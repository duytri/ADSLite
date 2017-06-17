package main.scala.obj

import org.apache.spark.mllib.linalg.{ Matrices, Matrix, Vector, Vectors }
import org.apache.spark.mllib.util.{ Loader, Saveable }
import org.apache.spark.graphx.{ Edge, EdgeContext, Graph, VertexId }
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{ Row, SparkSession }
import org.apache.hadoop.fs.Path

import breeze.linalg.{ argmax, argtopk, normalize, sum, DenseMatrix => BDM, DenseVector => BDV }
import breeze.numerics.{ exp, lgamma }

import org.json4s.DefaultFormats
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._

import main.scala.helper.Utils
import main.scala.helper.BPQ
import main.scala.helper.ADSOptimizer

abstract class Model {

  /** Number of hidden topics */
  def k: Int

  /** Number of all topics */
  def t: Int

  /** Vocabulary size (number of terms or terms in the vocabulary) */
  def vocabSize: Int

  /**
   * Concentration parameter (commonly named "alpha") for the prior placed on documents'
   * distributions over topics ("theta").
   *
   * This is the parameter to a Dirichlet distribution.
   */
  def docConcentration: Vector

  /**
   * Concentration parameter (commonly named "beta" or "eta") for the prior placed on topics'
   * distributions over terms.
   *
   * This is the parameter to a symmetric Dirichlet distribution.
   *
   * @note The topics' distributions over terms are called "beta" in the original LDA paper
   * by Blei et al., but are called "phi" in many later papers such as Asuncion et al., 2009.
   */
  def topicConcentration: Double

  /**
   * Return the topics described by weighted terms.
   *
   * @param maxTermsPerTopic  Maximum number of terms to collect for each topic.
   * @param eta  Topics' distributions over terms.
   * @param termSize  Actual terms size.
   * @return  Array over topics.  Each topic is represented as a pair of matching arrays:
   *          (term indices, term weights in topic).
   *          Each topic's terms are sorted in order of decreasing weight.
   */
  def describeTopics(maxTermsPerTopic: Int): Array[Array[(Int, Double)]]

  /**
   * Return the topics described by weighted terms.
   *
   * WARNING: If vocabSize and k are large, this can return a large object!
   *
   * @return  Array over topics.  Each topic is represented as a pair of matching arrays:
   *          (term indices, term weights in topic).
   *          Each topic's terms are sorted in order of decreasing weight.
   */
  def describeTopics(): Array[Array[(Int, Double)]] = describeTopics(vocabSize)
}

/**
 * Distributed LDA model.
 * This model stores the inferred topics, the full training dataset, and the topic distributions.
 */
class LDAModel(
  val graph: Graph[LDA.TopicCounts, LDA.TokenCount],
  val globalTopicTotals: LDA.TopicCounts,
  val k: Int,
  val t: Int,
  val vocabSize: Int,
  override val docConcentration: Vector,
  override val topicConcentration: Double,
  val iterationTimes: Array[Double])
    extends Model {

  import LDA._

  override def describeTopics(maxTermsPerTopic: Int): Array[Array[(Int, Double)]] = {
    val numTopics = k
    val phi = computePhi
    val result = Array.ofDim[(Int, Double)](k, maxTermsPerTopic)
    for (topic <- 0 until k) {
      val maxVertexPerTopic = phi.filter(_._1 == topic).takeOrdered(maxTermsPerTopic)(Ordering[Double].reverse.on(_._3))
      result(topic) = maxVertexPerTopic.map {
        case (topicId, termId, phi) =>
          (index2term(termId), phi)
      }
    }
    return result
  }

  def computeTheta(): RDD[(VertexId, Int, Double)] = {
    val alpha = this.docConcentration(0)
    graph.vertices.filter(LDA.isDocumentVertex).flatMap {
      case (docId, topicCounts) =>
        topicCounts.mapPairs {
          case (topicId, wordCounts) =>
            val thetaMK = ((wordCounts + alpha) / (topicCounts.data.sum + topicCounts.length * alpha))
            (docId, topicId, thetaMK)
        }.toArray
    }
  }

  def computePhi(): RDD[(Int, VertexId, Double)] = {
    val eta = this.topicConcentration
    val wordTopicCounts = this.globalTopicTotals
    val vocabSize = this.vocabSize
    graph.vertices.filter(LDA.isTermVertex).flatMap {
      case (termId, topicCounts) =>
        topicCounts.mapPairs {
          case (topicId, wordCounts) =>
            var phiKW = 0d
            if (topicId < k) {
              phiKW = ((wordCounts + eta) / (wordTopicCounts.data(topicId) + vocabSize * eta))
            } else {
              phiKW = ((wordCounts + eta) / (wordTopicCounts.data(topicId) + vocabSize * eta))
            }
            (topicId, termId, phiKW)
        }.toArray
    }
  }

  def computePerplexity(tokenSize: Long): Double = {
    val alpha = this.docConcentration(0)
    val eta = this.topicConcentration
    val wordTopicCounts = this.globalTopicTotals
    val vocabSize = this.vocabSize
    val sendMsg: EdgeContext[TopicCounts, TokenCount, Double] => Unit = (edgeContext) => {
      var thetaDotPhi = 0d
      edgeContext.srcAttr.activeIterator.foreach { // in a doc, foreach topic and number of words assigned to
        case (topicId1, wordCount) =>
          edgeContext.dstAttr.activeIterator.filter(_._1 == topicId1).foreach { // in a word, for each topic and number of instances assigned to
            case (topicId2, instanceCount) =>
              thetaDotPhi += ((wordCount + alpha) / (edgeContext.srcAttr.data.sum + edgeContext.srcAttr.length * alpha)) *
                ((instanceCount + eta) / (wordTopicCounts.data(topicId1) + vocabSize * eta))
          }
      }
      edgeContext.sendToDst(math.log(thetaDotPhi))
    }
    val mergMsg: (Double, Double) => Double =
      (a, b) =>
        a + b
    val docSum = graph.aggregateMessages[Double](sendMsg, _ + _) // mergMsg)
      .map(_._2).reduce(_ + _)
    return math.exp(-1 * docSum / tokenSize)
  }
}