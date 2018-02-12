
package xikai.minesweeper

import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.api.ndarray.INDArray
import org.deeplearning4j.nn.conf.{NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.conf.layers.{ConvolutionLayer, DenseLayer, OutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.ConvolutionMode
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator
import org.deeplearning4j.util.ModelSerializer

import java.io.File;

// Needed for stats to analyse the neural network.
/*
import org.deeplearning4j.ui.api.UIServer
import org.deeplearning4j.api.storage.StatsStorage
import org.deeplearning4j.ui.stats.StatsListener
import org.deeplearning4j.ui.storage.InMemoryStatsStorage
 */

case class NNPlayer(boardSize: Int, neuralNetwork: MultiLayerNetwork) {
  /* // Needed to analyse the neural network.
  val uiServer: UIServer = UIServer.getInstance() // For deeplearning4j network visualization
  val statsStorage: StatsStorage = new InMemoryStatsStorage()
  val listenerFrequency = 1
  neuralNetwork.setListeners(new StatsListener(statsStorage, listenerFrequency))
  uiServer.attach(statsStorage)
   */
  //Finally: open your browser and go to http://localhost:9000/train

  def train(input: Array[Array[Double]],
    label: Array[Array[Double]],
    batchSize: Int,
    epochs: Int
  ): Unit = {
    neuralNetwork.setInputMiniBatchSize(batchSize)

    val inputTrain:INDArray = Nd4j.create(input)
    val labelTrain:INDArray = Nd4j.create(label)

    val ds = new DataSet(inputTrain, labelTrain)
    //val dsIter: DataSetIterator = ds.iterateWithMiniBatches()

    println("Train model...")
    for (i <- 0 until epochs) {
      neuralNetwork.fit(new ListDataSetIterator(ds.batchBy(batchSize)))
    }
  }
  
  // filename should be a .zip file.
  def saveNetwork(filename: String) = {
    val locationToSave = new File(filename) //Where to save the network. Note: the file is in .zip format - can be opened externally
    val saveUpdater = true //Updater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this if you want to train your network more in the future
    ModelSerializer.writeModel(neuralNetwork, locationToSave, saveUpdater);
  }

  def avgErrorOnData(inputData: Array[Array[Double]],
    outputData: Array[Array[Double]]): Double = {
    var error: Double = 0.0
    for (i <- 0 until inputData.length) {
      val out: INDArray = neuralNetwork.output(Nd4j.create(inputData(i)))
      val n = outputData(i).length
      for (j <- 0 until n) {
        error += math.abs(outputData(i)(j) - out.getDouble(j))/n
      }
    }
    error
  }

  def output(gs: GameState): Vector[Vector[Double]] = {
    val n = gs.boardSize
    val out: INDArray = neuralNetwork.output(Nd4j.create(NNPlayer.gameStateToNNInput(gs)))
    for (j <- (0 until n).toVector) yield {
      for (i <- (0 until n).toVector) yield {
        out.getDouble(j*n + i)
      }
    }
  }

  def bestMove(gs: GameState, debug: Boolean = false): Option[Move] = {
    if (gs.isGameOver) None
    else {
      assert(gs.boardSize == boardSize,
        s"NNPlayer trained on size $boardSize, not ${gs.boardSize}")
      val n = boardSize
      val output: Vector[Vector[Double]] = this.output(gs)
      //println(output.map(_ mkString ", ").mkString("\n"))
      if (debug) { 
        println(output)
        println("Visible:")
        println(gs.visible)
      }


      var largestOutput: Double = -1.0
      var (iOfLargest, jOfLargest) = (-1, -1)
      for (j <- (0 until n);
        i <- (0 until n)) {
        if (!gs.visible(i)(j)) {
          val out = output(j)(i)
          if (out >= largestOutput) {
            largestOutput = out
            iOfLargest = i
            jOfLargest = j
          }
        }
      }
      //println(s"NN best move: ${(iOfLargest, jOfLargest)}")
      assert((iOfLargest, jOfLargest) != (-1, -1), s"NN played invalid move: $gs\n$output")
      Some((iOfLargest, jOfLargest))
    }
  }

  /**  
    * Return all positions from a game against itself.
    */
  def playAgainstSelf(debug: Boolean = false): Vector[GameState] = {
    def gamePositions(currGS: GameState, allPositions: Vector[GameState]): Vector[GameState] = {
      if (currGS.isGameOver) allPositions
      else {
        val move = bestMove(currGS, debug).get
        val nextGS: GameState = currGS.makeMove(move._1, move._2) match {
          case Right(gs) => gs
          case Left(error) =>
            throw new Exception(s"AI tries to play move: $move. But received error: $error")
        }
        gamePositions(nextGS, allPositions :+ nextGS)
      }
    }

    val gs = GameState.createNewGame(boardSize)
    gamePositions(gs, Vector(gs))
  }

  /**  Make this NN play against itself to generate training data.
    *  Returns a pair of array of inputs and array of outputs.
    *  @param size Number of game positions to generate training data from.
    */
  def createTrainingData(size: Int): (Array[Array[Double]], Array[Array[Double]]) = {
    //val inputs = Array.ofDim[Double](size, 11*boardSize*boardSize)
    //val outputs = Array.ofDim[Double](size, boardSize*boardSize)

    def positions(n: Int, pos: Vector[GameState]): Vector[GameState] = {
      if (n <= 0) pos
      else {
        //val posInSingleGame = playAgainstSelf()
        val posInSingleGame = Player.playSingleGame(boardSize, Player.randomMovePlayer, NNPlayer.toPlayer(this))
        // Drop last position, since the end position has no further moves.
        positions(
          n - (posInSingleGame.length - 2),
          pos ++ posInSingleGame.slice(1,math.min(size, posInSingleGame.length-1))
        )
      }
    }

    val games = positions(size, Vector[GameState]())
    val inputs = games.map(gs => NNPlayer.gameStateToNNInput(gs)).toArray
    val outputs = games.map(gs => NNPlayer.gameStateToNNOutput(gs)).toArray
    (inputs, outputs)
  }
}

object NNPlayer {
  def untrainedPlayer(boardSize: Int, 
                      seed: Int, 
                      iter: Int, 
                      learningRate: Double): NNPlayer = {

    //val inputNum: Int = 11*boardSize*boardSize
    val inputNum: Int = 5*boardSize*boardSize
    val outputNum: Int = boardSize*boardSize
    val nChannels: Int = 5

    val conf = new NeuralNetConfiguration.Builder()
      .seed(seed)
      .iterations(iter)
      .miniBatch(true)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .learningRate(learningRate)
      .updater(Updater.NESTEROVS)
      .momentum(0.9)
      .regularization(true).l2(learningRate*.01)
      .list
      .layer(0, new ConvolutionLayer.Builder(3,3)
        .nIn(nChannels)
        .stride(1,1)
        .nOut(20) // 200 // 20
        .convolutionMode(ConvolutionMode.Same)
        .activation(Activation.RELU)
        .build)
      .layer(1, new ConvolutionLayer.Builder(3,3)
        .stride(1,1)
        .nOut(30) // 100 // 30
        .convolutionMode(ConvolutionMode.Same)
        .activation(Activation.RELU)
        .build)
      .layer(2, new ConvolutionLayer.Builder(3,3)
        .stride(1,1)
        .nOut(50) // 50
        .convolutionMode(ConvolutionMode.Same)
        .activation(Activation.RELU)
        .build)
    
      .layer(3, new ConvolutionLayer.Builder(3,3)
        .stride(1,1)
        .nOut(2) // 10 // 2
        .convolutionMode(ConvolutionMode.Same)
        .activation(Activation.RELU)
        .build)
    
      .layer(4, new OutputLayer.Builder(LossFunctions.LossFunction.EXPLL)
        .activation(Activation.SIGMOID)
        .nOut(outputNum)
        .build)
      .setInputType(InputType.convolutionalFlat(boardSize, boardSize, nChannels))
      .pretrain(false).backprop(true)
      .build

    val model = new MultiLayerNetwork(conf)
    model.init()
    model.setListeners(new ScoreIterationListener(5))

    NNPlayer(boardSize, model)
  }

  def gameStateToNNInput(gs: GameState): Array[Double] = {
    val n = gs.boardSize
    //val input = new Array[Double](11 * n * n)
    val input = new Array[Double](5 * n * n)

    for (j <- 0 until n;
         i <- 0 until n) {
      val index = j*n + i
      input(index + 0) = if (gs.visible(i)(j)) 1.0 else 0.0
      input(index + 1*n*n) = if (gs.visible(i)(j) && gs.board(i)(j) == FLAG) 1.0 else 0.0

      if (gs.visible(i)(j) && gs.board(i)(j) != FLAG) {
        //input(index + 2 + gs.board(i)(j)) = 1.0
        input(index + 2*n*n) = gs.board(i)(j) / 8.0
      }

      if ((i == 0 && j == 0) || (i == n-1 && j == 0) ||
        (i == 0 && j == n-1) || (i == n-1 && j == n-1)) { // Corner cell
        input(index + 3*n*n) = 1.0
      } else if (i == 0 || i == n-1 || j == 0 || j == n-1) { // Border
        input(index + 4*n*n) = 1.0
      }
    }

    input
  }

  def gameStateToNNOutput(gs: GameState): Array[Double] = {
    val n = gs.boardSize
    val output = new Array[Double](n * n)

    for (j <- 0 until n;
         i <- 0 until n) {
      val index = j*n + i
      output(index) = if (gs.board(i)(j) == FLAG) 1.0 else 0.0
    }
    output
  }

  def gameStateToTrainingData(gs: GameState): (Array[Double], Array[Double]) = {
    val input = gameStateToNNInput(gs)
    val output = gameStateToNNOutput(gs)
    (input, output)
  }

  def toPlayer(player: NNPlayer): Player[NNPlayer] = new Player[NNPlayer] {
    def bestMove(gs: GameState): Option[Move] = player.bestMove(gs)
  }

  def loadNNPlayer(boardSize: Int, filename: String): NNPlayer = {
    val restored: MultiLayerNetwork = ModelSerializer.restoreMultiLayerNetwork(filename)
    NNPlayer(boardSize, restored)
  }
}

