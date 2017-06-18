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
import main.scala.obj.LDA.TopicCounts

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
  val documents: RDD[(Long, TopicCounts, Array[(Long, TopicCounts)])],
  val globalTopicTotals: LDA.TopicCounts,
  val k: Int,
  val t: Int,
  val vocabSize: Int,
  override val docConcentration: Vector,
  override val topicConcentration: Double)
    extends Model {

  import LDA._

  override def describeTopics(maxTermsPerTopic: Int): Array[Array[(Int, Double)]] = {
    val numTopics = k
    val phi = computePhi
    val result = Array.ofDim[(Int, Double)](k, maxTermsPerTopic)
    for (topic <- 0 until k) {
      val maxTermPerTopic = phi.filter(_._1 == topic).takeOrdered(maxTermsPerTopic)(Ordering[Double].reverse.on(_._3))
      result(topic) = maxTermPerTopic.map {
        case (topicId, termId, phi) =>
          (index2term(termId), phi)
      }
    }
    return result
  }

  def computeTheta(): RDD[(Long, Int, Double)] = {
    val alpha = this.docConcentration(0)
    val numTopics = this.t
    documents.flatMap {
      case (docId, docTopicCounts, termArray) =>
        docTopicCounts.activeIterator.map {
          case (topicId, termCounts) =>
            (docId, topicId, (termCounts + alpha) / (docTopicCounts.reduce(_ + _) + numTopics * alpha))
        }
    }
  }

  def computePhi(): RDD[(Int, Long, Double)] = {
    val V = vocabSize
    val eta = this.topicConcentration
    val wordTopicCounts = this.globalTopicTotals
    documents.flatMap(_._3).reduceByKey(_ + _).flatMap { // sum all words in corpus and flatMap
      case (wordId, topicCounts) =>
        topicCounts.activeIterator.map {
          case (topicId, termCounts) =>
            (topicId, wordId, ((termCounts + eta) / (wordTopicCounts(topicId) + V * eta)))
        }
    }
  }

  def computePerplexity(tokenSize: Double): Double = {
    val alpha = this.docConcentration(0)
    val numTopics = this.t
    val V = vocabSize
    val eta = this.topicConcentration
    val wordTopicCounts = this.globalTopicTotals
    var docSum = documents.map {
      case (docId, docTopicCounts, termArray) => {
        var wordSum = termArray.map {
          case (wordId, termTopicCount) => {
            var topicSum = termTopicCount.activeIterator.map {
              case (topicId, termCounts) => {
                termCounts * (termCounts + alpha) / (docTopicCounts.reduce(_ + _) + numTopics * alpha) * (termCounts + eta) / (wordTopicCounts(topicId) + V * eta)
              }
            }.fold(0d)(_ + _)
            math.log(topicSum)
          }
        }.fold(0d)(_ + _)
        wordSum
      }
    }.fold(0d)(_ + _)
    return math.exp(-1 * docSum / tokenSize)
  }
}