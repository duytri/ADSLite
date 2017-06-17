package main.scala

import main.scala.obj.Document
import main.scala.helper.ADSCmdOption
import main.java.commons.cli.MissingOptionException
import main.java.commons.cli.MissingArgumentException
import main.java.commons.cli.CommandLine
import main.java.commons.cli.UnrecognizedOptionException
import main.scala.obj.Parameter
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import java.io.File
import breeze.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import scala.collection.mutable.ArrayBuffer
import main.scala.helper.Utils
import main.scala.obj.LDAModel
import main.scala.obj.LDA
import main.scala.connector.File2KS
import org.apache.spark.sql.SparkSession

object ADSLite {

  def main(args: Array[String]): Unit = {
    println("#################### Gibbs sampling ADS-LDA in Apache Spark ####################")
    try {
      var cmd = ADSCmdOption.getArguments(args)
      if (cmd.hasOption("help")) {
        ADSCmdOption.showHelp()
      } else {
        // set user parameters
        var params = new Parameter
        params.getParams(cmd)
        if (!params.checkRequirement) {
          println("ERROR!!! Phai nhap day du cac tham so: alpha, beta, directory, ksource, ntopics, niters")
          ADSCmdOption.showHelp()
          return
        } else {
          //~~~~~~~~~~~ Spark ~~~~~~~~~~~
          val conf = new SparkConf().setAppName("ADS-LDA").setMaster("spark://PTNHTTT05:7077")
          val spark = SparkSession.builder().config(conf).getOrCreate()
          val sc =  spark.sparkContext

          //~~~~~~~~~~~ Body ~~~~~~~~~~~
          // Load documents, and prepare them for LDA.
          val preprocessStart = System.nanoTime()
          val (corpus, vocabArray, actualNumTokens) = Utils.preprocess(spark, params.directory + "/*")
          corpus.cache()
          val actualCorpusSize = corpus.count()
          val actualVocabSize = vocabArray.length
          val knowledge: Array[Array[(Int, Int)]] = File2KS.readKnowledgeSrc(params.ks, vocabArray)
          val preprocessElapsed = (System.nanoTime() - preprocessStart) / 1e9

          println()
          println(s"Corpus summary:")
          println(s"\t Training set size: $actualCorpusSize documents")
          println(s"\t Vocabulary size: $actualVocabSize terms")
          println(s"\t Training set size: $actualNumTokens tokens")
          println(s"\t Preprocessing time: $preprocessElapsed sec")
          println()

          // Cluster the documents into three topics using LDA
          val adsLDA = new LDA()
            .setK(params.K)
            .setAlpha(params.alpha)
            .setBeta(params.beta)
            .setMaxIterations(params.niters)
            .setKnowledge(knowledge)

          val startTime = System.nanoTime()
          // Estimate
          val ldaModel = adsLDA.run(corpus, knowledge, actualVocabSize)

          val elapsed = (System.nanoTime() - startTime) / 1e6

          println(s"Finished training LDA model.  Summary:")
          val millis = (elapsed % 1000).toInt
          val seconds = ((elapsed / 1000) % 60).toInt
          val minutes = ((elapsed / (1000 * 60)) % 60).toInt
          val hours = ((elapsed / (1000 * 60 * 60)) % 24).toInt
          println(s"\t Training time: $hours hour(s) $minutes minute(s) $seconds second(s) and $millis milliseconds")

          if (ldaModel.isInstanceOf[LDAModel]) {
            val distLDAModel = ldaModel.asInstanceOf[LDAModel]
            val perplexity = distLDAModel.computePerplexity(actualNumTokens)
            println(s"\t Training data perplexity: $perplexity")
            println()
          }

          // Print the topics, showing the top-weighted terms for each topic.
          val topicIndices = ldaModel.describeTopics(params.twords)
          val topics = topicIndices.map(topic => {
            topic.map {
              case (termIndex, weight) =>
                (vocabArray(termIndex), weight)
            }
          })
          println(s"${params.K} topics:")
          topics.zipWithIndex.foreach {
            case (topic, i) =>
              println(s"---------- TOPIC $i ---------")
              topic.foreach {
                case (term, weight) =>
                  println(s"$term\t$weight")
              }
              println("---------------------------\n")
          }

          //ldaModel.countGraphInfo()

          sc.stop()
          //spark.stop()

        }
      }
    } catch {
      case moe: MissingOptionException => {
        println("ERROR!!! Phai nhap day du cac tham so: alpha, beta, directory, ksource, ntopics, niters")
        ADSCmdOption.showHelp()
      }
      case mae: MissingArgumentException => {
        mae.printStackTrace()
        println("ERROR!!! Thieu gia tri cua cac tham so.")
        ADSCmdOption.showHelp()
      }
      case uoe: UnrecognizedOptionException => {
        uoe.printStackTrace()
        println("ERROR!!! Chuong trinh khong ho tro tham so ban da nhap.")
        ADSCmdOption.showHelp()
      }
      case e: Throwable => e.printStackTrace()
    }
  }
}
