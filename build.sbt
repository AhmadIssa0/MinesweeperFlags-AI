
name := "Minesweeper Flags"

//scalaVersion := "2.12"
fork := true
javaOptions += "-Xmx3G"

// Add dependency on ScalaFX library
libraryDependencies += "org.scalafx" %% "scalafx" % "12.0.2-R18"

// Determine OS version of JavaFX binaries
lazy val osName = System.getProperty("os.name") match {
  case n if n.startsWith("Linux")   => "linux"
  case n if n.startsWith("Mac")     => "mac"
  case n if n.startsWith("Windows") => "win"
  case _ => throw new Exception("Unknown platform!")
}

// Add dependency on JavaFX libraries, OS dependent
lazy val javaFXModules = Seq("base", "controls", "fxml", "graphics", "media", "swing", "web")
libraryDependencies ++= javaFXModules.map( m =>
  "org.openjfx" % s"javafx-$m" % "12.0.2" classifier osName
)
//libraryDependencies += "org.scalafx" %% "scalafx" % "8.0.92-R10"
//libraryDependencies += "org.scalafx" %% "scalafx" % "8.0.144-R12"
libraryDependencies += "org.scalafx" %% "scalafx" % "12.0.2-R18"
//libraryDependencies += "org.scalafx" %% "scalafx" % "2.2.76-R11"

//libraryDependencies += "org.deeplearning4j" % "deeplearning4j-core" % "0.9.1"//non-cuda
//libraryDependencies += "org.nd4j" % "nd4j-native-platform" % "0.9.1" //non-cuda

libraryDependencies += "org.deeplearning4j" % "deeplearning4j-core" % "1.0.0-beta7"//non-cuda
libraryDependencies += "org.nd4j" % "nd4j-native-platform" % "1.0.0-beta7" //non-cuda

//libraryDependencies += "org.deeplearning4j" % "deeplearning4j-cuda-10.2" % "1.0.0-beta7"
//libraryDependencies += "org.nd4j" % "nd4j-cuda-10.2-platform" % "1.0.0-beta7"

/***
libraryDependencies += "org.deeplearning4j" % "deeplearning4j-ui_2.11" % "0.9.1"
libraryDependencies += "org.bytedeco.javacpp-presets" % "cuda" % "8.0-6.0-1.3"
//libraryDependencies += "org.bytedeco.javacpp-presets" % "cuda" % "8.0-6.0-1.4.1"

//libraryDependencies += "org.deeplearning4j" % "deeplearning4j-core" % "0.9.1"
libraryDependencies += "org.deeplearning4j" % "deeplearning4j-cuda-8.0" % "0.9.1"
libraryDependencies += "org.nd4j" % "nd4j-cuda-8.0-platform" % "0.9.1"
 ***/
//libraryDependencies += "org.nd4j" % "nd4j-api" % "0.9.1"

//assemblyJarName in assembly := "Minesweeper.jar"

/*
retrieveManaged := true
test in assembly := {}

assemblyMergeStrategy in assembly := {
   case PathList("META-INF", xs @ _*) => MergeStrategy.discard
    case x => MergeStrategy.first
}

artifact in (Compile, assembly) := {
    val art = (artifact in (Compile, assembly)).value
      art.copy(`classifier` = Some("assembly"))
}

addArtifact(artifact in (Compile, assembly), assembly)

cancelable in Global := true
 */
