package main.scala.connector

import java.io.File
import java.io.BufferedReader
import java.io.FileReader
import java.util.StringTokenizer
import scala.collection.mutable.ArrayBuffer

object File2KS {
  def readKnowledgeSrc(filename: String, vocabArray: Array[String]): Array[Array[(Int, Int)]] = {
    val file = new File(filename)
    val reader = new BufferedReader(new FileReader(file))
    var line = reader.readLine()
    var resBuffer = new ArrayBuffer[Array[(Int, Int)]]
    while (line != null) {
      val tknr = new StringTokenizer(line, " ")
      val str_frq = new ArrayBuffer[(Int, Int)]((tknr.countTokens() - 1) / 2)
      val topic = tknr.nextToken()
      while (tknr.hasMoreTokens()) {
        str_frq.append((vocabArray.indexOf(tknr.nextToken()), tknr.nextToken().toInt))
      }
      resBuffer.append(str_frq.toArray)
      line = reader.readLine()
    }
    resBuffer.toArray
  }
}