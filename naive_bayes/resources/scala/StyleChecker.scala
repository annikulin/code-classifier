/**
 * Copyright (c) 2014 MongoDB, Inc.
 *
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 *
 * For questions and comments about this product, please see the project page at:
 *
 * https://github.com/mongodb/mongo-scala-driver
 *
 */
import sbt.File
import java.io.ByteArrayOutputStream
import java.io.PrintStream
import org.scalastyle._

object StyleChecker {
  val maxResult = 100

  class CustomTextOutput[T <: FileSpec]() extends Output[T] {
    private val messageHelper = new MessageHelper(this.getClass().getClassLoader())

    var fileCount: Int = _
    override def message(m: Message[T]): Unit = m match {
      case StartWork() =>
      case EndWork() =>
      case StartFile(file) =>
        print("Checking file " + file + "...")
        fileCount = 0
      case EndFile(file) =>
        if (fileCount == 0) println(" OK!")
      case StyleError(file, clazz, key, level, args, line, column, customMessage) =>
        report(line, column, messageHelper.text(level.name),
               Output.findMessage(messageHelper, clazz, key, args, customMessage))
      case StyleException(file, clazz, message, stacktrace, line, column) =>
        report(line, column, "error", message)
    }

    private def report(line: Option[Int], column: Option[Int], level: String, message: String) {
      if (fileCount == 0) println("")
      fileCount += 1
      println("  " + fileCount + ". " + level + pos(line, column) + ":")
      println("     " + message)
    }

    private def pos(line: Option[Int], column: Option[Int]): String = line match {
      case Some(line) => " at line " + line + (column match {
        case Some(column) => " character " + column
        case None => ""
      })
      case None => ""
    }
  }

  def score(outputResult: OutputResult) = {
    val penalties = outputResult.errors + outputResult.warnings
    scala.math.max(maxResult - penalties, 0)
  }

  def assess(sources: Seq[File]): (String, Int) = {

    val configFile = new File("./project/scalastyle-config.xml").getAbsolutePath

    val messages = new ScalastyleChecker().checkFiles(
      ScalastyleConfiguration.readFromXml(configFile),
      Directory.getFiles(None, sources))

    val output = new ByteArrayOutputStream()
    val outputResult = Console.withOut(new PrintStream(output)) {
      new CustomTextOutput().output(messages)
    }

    val msg =
      output.toString +
      "Processed " + outputResult.files + " file(s)\n" +
      "Found " + outputResult.errors + " errors\n" +
      "Found " + outputResult.warnings + " warnings\n" +
      (if (outputResult.errors+outputResult.warnings > 0) "Consult the scala style guide at http://www.scalastyle.org/" else "")

    (msg, score(outputResult))
  }
}
