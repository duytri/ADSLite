package main.scala.helper

import org.apache.spark.graphx._
import main.scala.obj.LDA.{ TopicCounts, TokenCount }
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{ DenseVector, Matrices, SparseVector, Vector, Vectors }
import main.scala.obj.LDA
import breeze.linalg.{ all, normalize, sum, DenseMatrix => BDM, DenseVector => BDV }
import scala.util.Random
import main.scala.obj.LDAModel
import main.scala.obj.Model
import scala.collection.mutable.ArrayBuffer

class ADSOptimizer {
  import LDA._

  // The following fields will only be initialized through the initialize() method
  var graph: Graph[TopicCounts, TokenCount] = null
  var k: Int = 0
  var t: Int = 0
  var vocabSize: Int = 0
  var docConcentration: Double = 0
  var topicConcentration: Double = 0

  var knowledge: Array[Array[(Int, Int)]] = null
  var deltaPow: Array[Array[(Int, Double)]] = null
  var deltaPowSum: Array[Double] = null

  var checkpointInterval: Int = 10

  var graphCheckpointer: PeriodicGraphCheckpointer[TopicCounts, TokenCount] = null

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
      for (wordId <- 0 until deltaTopic.length) {
        rowBuff.append((deltaTopic(wordId)._1, math.pow(deltaTopic(wordId)._2, Utils.lamda)))
        sumBuff += rowBuff(wordId)._2
      }
      deltaPow(topic) = rowBuff.toArray
      deltaPowSum(topic) = sumBuff
    }

    this.docConcentration = if (docConcentration <= 0) 50.0 / t else docConcentration
    this.topicConcentration = if (topicConcentration <= 0) 1.1 else topicConcentration
    val randomSeed = lda.getSeed

    // For each document, create an edge (Document -> Term) for each unique term in the document.
    val edges: RDD[Edge[TokenCount]] = docs.flatMap {
      case (docID: Long, termCounts: Vector) =>
        // Add edges for terms with non-zero counts.
        Utils.asBreeze(termCounts).activeIterator.filter(_._2 != 0.0).map {
          case (term, cnt) =>
            Edge(docID, term2index(term), cnt)
        }
    }

    // Create vertices.
    // Initially, we use random soft assignments of tokens to topics (random gamma).
    val docTermVertices: RDD[(VertexId, TopicCounts)] = {
      val verticesTMP: RDD[(VertexId, TopicCounts)] =
        edges.flatMap { edge =>
          val gamma = Utils.randomVectorInt(t, edge.attr.toInt)
          Seq((edge.srcId, gamma), (edge.dstId, gamma))
        }
      verticesTMP
    }

    // Partition such that edges are grouped by document
    this.graph = Graph(docTermVertices, edges).partitionBy(PartitionStrategy.EdgePartition1D)
    this.k = k
    this.t = t
    this.vocabSize = docs.take(1).head._2.size
    this.globalTopicTotals = computeGlobalTopicTotals()
    this
  }

  def next(): ADSOptimizer = {
    require(graph != null, "graph is null, EMLDAOptimizer not initialized.")

    val hiddenTopic = k
    val eta = topicConcentration
    val W = vocabSize
    val alpha = docConcentration
    val N_k = globalTopicTotals
    val dPow = deltaPow
    val dPowSum = deltaPowSum
    val vcbSize = vocabSize
    val sendMsg: EdgeContext[TopicCounts, TokenCount, (Boolean, TopicCounts)] => Unit =
      (edgeContext) => {
        // Compute N_{wj} gamma_{wjk}
        val N_wj = edgeContext.attr
        // E-STEP: Compute gamma_{wjk} (smoothed topic distributions), scaled by token count
        // N_{wj}.
        val scaledTopicDistribution: TopicCounts =
          computePTopic(LDA.index2term(edgeContext.dstId), edgeContext.srcAttr, edgeContext.dstAttr, N_k, W, hiddenTopic, eta, alpha, dPow, dPowSum)
        edgeContext.sendToDst((false, edgeContext.dstAttr + scaledTopicDistribution))
        edgeContext.sendToSrc((false, edgeContext.srcAttr + scaledTopicDistribution))
      }
    // The Boolean is a hack to detect whether we could modify the values in-place.
    // TODO: Add zero/seqOp/combOp option to aggregateMessages. (SPARK-5438)
    val mergeMsg: ((Boolean, TopicCounts), (Boolean, TopicCounts)) => (Boolean, TopicCounts) =
      (m0, m1) => {
        val sum =
          if (m0._1) {
            m0._2 += m1._2
          } else if (m1._1) {
            m1._2 += m0._2
          } else {
            m0._2 + m1._2
          }
        (true, sum)
      }
    // M-STEP: Aggregation computes new N_{kj}, N_{wk} counts.
    val docTopicDistributions: VertexRDD[TopicCounts] =
      graph.aggregateMessages[(Boolean, TopicCounts)](sendMsg, mergeMsg)
        .mapValues(_._2)
    // Update the vertex descriptors with the new counts.
    val newGraph = Graph(docTopicDistributions, graph.edges)
    graph = newGraph
    globalTopicTotals = computeGlobalTopicTotals()
    this
  }

  /**
   * Aggregate distributions over topics from all term vertices.
   *
   * Note: This executes an action on the graph RDDs.
   */
  var globalTopicTotals: TopicCounts = null

  private def computeGlobalTopicTotals(): TopicCounts = {
    graph.vertices.filter(isDocumentVertex).values.reduce(_ + _)
  }

  def getADSModel(iterationTimes: Array[Double]): LDAModel = {
    require(graph != null, "graph is null, ADSOptimizer not initialized.")
    // The constructor's default arguments assume gammaShape = 100 to ensure equivalence in
    // LDAModel.toLocal conversion.
    new LDAModel(this.graph, this.globalTopicTotals, this.k, this.t, this.vocabSize,
      Vectors.dense(Array.fill(this.t)(this.docConcentration)), this.topicConcentration,
      iterationTimes)
  }
}