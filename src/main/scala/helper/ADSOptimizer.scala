package main.scala.helper

import org.apache.spark.graphx._
import main.scala.obj.LDA.{ TopicCounts, TokenCount }
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{ DenseVector, Matrices, SparseVector, Vector, Vectors }
import main.scala.obj.LDA
import breeze.linalg.{ all, normalize, sum, DenseMatrix => BDM, DenseVector => BDV }
import main.scala.obj.LDAModel
import main.scala.obj.Model
import scala.collection.mutable.ArrayBuffer
import org.dmg.pmml.False
import org.apache.spark.storage.StorageLevel

class ADSOptimizer {
  import LDA._

  // The following fields will only be initialized through the initialize() method
  var documents: RDD[(Long, TopicCounts, Array[(Long, TopicCounts)])] = null
  //var docDict: Array[(Long, Vector)] = null
  var k: Int = 0
  var t: Int = 0
  var vocabSize: Int = 0
  var docConcentration: Double = 0
  var topicConcentration: Double = 0

  var knowledge: Array[Array[(Int, Int)]] = null
  var deltaPow: Array[Array[(Int, Double)]] = null
  var deltaPowSum: Array[Double] = null
  /**
   * Compute bipartite term/doc graph.
   */
  def initialize(
    docs: RDD[(Long, Vector)],
    knowledge: Array[Array[(Int, Int)]],
    vocabSize: Long,
    lda: LDA): ADSOptimizer = {
    // LDAOptimizer currently only supports symmetric document-topic priors
    val docConcentration = lda.getDocConcentration

    val topicConcentration = lda.getTopicConcentration
    val k = lda.getK
    val t = lda.getT
    val b = t - k

    this.knowledge = knowledge

    this.deltaPow = Array.ofDim[Array[(Int, Double)]](b)
    this.deltaPowSum = Array.ofDim[Double](b)

    for (topic <- 0 until b) {
      val deltaTopic = knowledge(topic) // get frequent of word
      var rowBuff = new ArrayBuffer[(Int, Double)]
      var sumBuff = 0.0
      deltaTopic.filter(_._1 >= 0).foreach {
        case (wordId, frequent) =>
          val pow = math.pow(frequent, Utils.lamda)
          rowBuff.append((wordId, pow))
          sumBuff += pow
      }
      deltaPow(topic) = rowBuff.toArray
      deltaPowSum(topic) = sumBuff
    }

    this.docConcentration = if (docConcentration <= 0) 50.0 / t else docConcentration
    this.topicConcentration = if (topicConcentration <= 0) 1.1 else topicConcentration
    val randomSeed = lda.getSeed

    this.documents = docs.map {
      case (docId: Long, termCounts: Vector) => {
        // Add edges for terms with non-zero counts.
        val iterCounts = Utils.asBreeze(termCounts).activeIterator.filter(_._2 != 0.0)
        var termPart = iterCounts.map {
          case (term, cnt) => {
            val gamma = Utils.randomVectorInt(t, cnt.toInt)
            (term2index(term), gamma)
          }
        }.toArray
        val docAssign = termPart.map(_._2).reduce(_ + _)
        (docId, docAssign, termPart)
      }
    }

    this.k = k
    this.t = t
    this.vocabSize = vocabSize.toInt
    this.globalTopicTotals = computeGlobalTopicTotals()
    this
  }

  def next(): ADSOptimizer = {

    val hiddenTopic = k
    val eta = topicConcentration
    val W = vocabSize
    val alpha = docConcentration
    val N_k = globalTopicTotals
    val dPow = deltaPow
    val dPowSum = deltaPowSum

    // re-sampling
    val newDocument = documents.map {
      case (docId, docTopicCounts, termArray) => {
        var deltaDocTopicCnt = new TopicCounts(docTopicCounts.length)
        val newTermArray = termArray.map {
          case (termId, termTopicCounts) => {
            val deltaTopic = computePTopic(termId.toInt, docTopicCounts, termTopicCounts, N_k, W, hiddenTopic, eta, alpha, dPow, dPowSum)
            deltaDocTopicCnt += deltaTopic
            (termId, termTopicCounts + deltaTopic)
          }
        }
        (docId, docTopicCounts + deltaDocTopicCnt, newTermArray, deltaDocTopicCnt)
      }
    }

    this.documents = newDocument.map(x => (x._1, x._2, x._3))
    this.documents.persist(StorageLevel.MEMORY_AND_DISK)
    // re-count global topic assignment
    globalTopicTotals += newDocument.map(_._4).reduce(_+_)
    newDocument.unpersist(false)
    this
  }

  /**
   * Aggregate distributions over topics from all term vertices.
   *
   * Note: This executes an action on the graph RDDs.
   */
  var globalTopicTotals: TopicCounts = null

  private def computeGlobalTopicTotals(): TopicCounts = {
    documents.map(_._2).reduce(_ + _)
  }

  def getADSModel(): LDAModel = {
    require(documents != null, "graph is null, ADSOptimizer not initialized.")
    // The constructor's default arguments assume gammaShape = 100 to ensure equivalence in
    // LDAModel.toLocal conversion.
    new LDAModel(this.documents, this.globalTopicTotals, this.k, this.t, this.vocabSize,
      Vectors.dense(Array.fill(this.t)(this.docConcentration)), this.topicConcentration)
  }
}