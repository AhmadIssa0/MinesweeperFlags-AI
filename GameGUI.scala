package xikai.minesweeper

import scalafx.Includes._
import scalafx.animation.AnimationTimer
import scalafx.application.JFXApp
import scalafx.application.JFXApp.PrimaryStage
import scalafx.event._
import scalafx.scene.effect._
import scalafx.scene.Group
import scalafx.scene.input._
import scalafx.scene.image.Image
import scalafx.scene.image.ImageView
import scalafx.scene.paint.Color._
import scalafx.scene.Scene
import scalafx.scene.shape.Rectangle
import scalafx.scene.text.Text
import scalafx.scene.control._
import scalafx.scene.layout._
import scalafx.geometry._

/** TODO: 
  * 1) Fix resizing the window.
  * 2) Replace the grid button text with images.
  *
  */
class GameGUI[A](nWidth: Int, nHeight: Int, dWidth: Double, dHeight: Double, aiPlayer: Player[A]) extends JFXApp {
  /**  GameGUI is a subclass of JFXApp.
    *  JFXApp contains a variable `stage: PrimaryStage`
    *  which we set. The PrimaryStage is the top level window.
    */
  var gameState = GameState.createNewGame(nWidth)
  val (grid, gridPane) = createGridPaneAndToggleButtons() 

  // Type alias for the buttons representing our grid.
  type GridToggleButtons = Vector[Vector[ToggleButton]]

  /**  Return a pair of:
    *    GridToggleButtons representings the squares of the minesweeper grid.
    *    GridPane which is a layout manager organising our buttons into a grid on the screen.
    *  TODO: Set the buttons to be nice images which can represent mines etc.
    */
  def createGridPaneAndToggleButtons(): (GridToggleButtons, GridPane) = {
    val gridPane = new GridPane() // Holds and organises our buttons in the window.
    gridPane.prefWidth = nWidth*dWidth
    gridPane.prefHeight = nHeight*dHeight

    // Set the width and height of each grid element to fill entire space.
    val colConstraint = new ColumnConstraints()
    colConstraint.setPercentWidth(dWidth)
    val rowConstraint = new RowConstraints()
    rowConstraint.setPercentHeight(dHeight)
    for (i <- 0 until nWidth) {
      gridPane.columnConstraints.add(colConstraint)
    }
    for (i <- 0 until nHeight) {
      gridPane.rowConstraints.add(rowConstraint)
    }

    // Create the buttons and add them to the grid pane
    val grid =
    for (x <- (0 until nWidth).toVector) yield {
      for (y <- (0 until nHeight).toVector) yield {
        val button = new ToggleButton("")
        button.alignment = Pos.CENTER

        // Make buttons fill the gridpane
        button.prefHeight = java.lang.Double.MAX_VALUE
        button.prefWidth = java.lang.Double.MAX_VALUE

        button.style = "-fx-border-color: #696969;"

        button
      }
    }

    for (x <- (0 until nWidth).toVector) yield {
      for (y <- (0 until nHeight).toVector) yield {
        grid(x)(y).onAction = (e: ActionEvent) => {
          if (! grid(x)(y).disabled.value && gameState.isBluesTurn) {
            makeMove(x, y)
            while (!gameState.isBluesTurn && !gameState.isGameOver) { // AI's move
              //Thread.sleep(1000)
              aiPlayer.bestMove(gameState) match {
                case Some((x, y)) => makeMove(x, y)
                case None => throw new Exception("AI couldn't make a valid move in this position: " + gameState)
              }
            }
          }
        }

        gridPane.add(grid(x)(y), x, y)
      }
    }
    gridPane.style = "-fx-border-width: 6;" + "-fx-border-color: #696969"
    (grid, gridPane)
  }

  def makeMove(x: Int, y: Int): Unit = {
    println("Making move: " + (x,y))
    gameState.makeMove(x, y) match {
      case Left(error) => println(error)
      case Right(newGameState) => {
        gameState = newGameState
        updateGameDisplay(grid, newGameState, Some((x, y)))
      }
    }
  }

  val colors = List("blue", "green", "red", "darkblue", "maroon", "orange", "purple", "hotpink")
  
  def updateSquare(x: Int, y: Int, grid: GridToggleButtons, gameState: GameState, lastMove: Option[Move]): Unit = {
    // Modify button grid(x)(y) to reflect the new state
    // state == MINE represents a mine
    // otherwise state will be an integer from 0 to 8 which counts
    // the number of neighbouring mines.

    val borderColor = if (!lastMove.isEmpty && lastMove.get._1 == x && lastMove.get._2 == y) "#ff3333" else "#696969"

    grid(x)(y).disable = gameState.visible(x)(y)
    gameState.board(x)(y) match {
      case FLAG => {
        val flagFilepath = 
          if (gameState.blueFlagsCaptured.contains (x ,y)) blueflagFilepath
          else redflagFilepath
          
        grid(x)(y).style = "-fx-background-image: url(" + flagFilepath + ");" +
          "-fx-background-size: 18px 18px;"+
          "-fx-background-repeat: no-repeat;"+
          "-fx-background-position: center;"+
          "-fx-opacity: 1.0;"+
          s"-fx-border-color: ${borderColor};"
      }
      case 0 => {
        grid(x)(y).style = "-fx-background-color: #eeffee;"+
        s"-fx-border-color: ${borderColor};"+
        "-fx-opacity: 1.0;"
        grid(x)(y).text = ""
      }
      case k => {
        grid(x)(y).text = k.toString
        val textColor = "-fx-text-fill: "+colors(k-1)+ ";"
        grid(x)(y).style = "-fx-background-color: #a9a9a9;"+
        s"-fx-border-color: ${borderColor};"+
        textColor +
        "-fx-opacity: 1.0;"
      }
    }
  }

  val redflagFilepath = new java.io.File("flag-red.png").toURI().toURL().toString()
  val blueflagFilepath = new java.io.File("flag-blue.png").toURI().toURL().toString()

  def updateGameDisplay(grid: GridToggleButtons, gameState: GameState, lastMove: Option[Move]): Unit = {
    // Update buttons to display game state.
    // Call updateSquare for every square.
    for (x <- (0 until gameState.boardSize).toVector) yield {
      for (y <- (0 until gameState.boardSize).toVector) yield {
        if (gameState.visible(x)(y)) {
          val state = gameState.board(x)(y)
          updateSquare(x, y, grid, gameState, lastMove)
        }
      }
    }
  }

  def resetGame(): Unit = {
    gameState = GameState.createNewGame(gameState.boardSize)
    val (gridNew, gridPaneNew) = createGridPaneAndToggleButtons()
  }

  /**  Set up the PrimaryStage (main window).
    *  We set the gridpane to be the contents of the scene,
    *  automatically adding all its children (the buttons).
    */
  stage = new PrimaryStage {
    title = "Minesweeper Flags"
    scene = new Scene(nWidth*dWidth, nHeight*dHeight + 2*dHeight) {
      //nWdith = # of grids in x direction
      //nHeight = # of grids in y direction
      val borderPane = new BorderPane()
      
      val menubar   = new MenuBar
      val gameMenu  = new Menu("Game")
      val startItem = new MenuItem("Restart")
      startItem.accelerator = new KeyCodeCombination(KeyCode.N, KeyCombination.ControlDown)
      val sizeItem  = new MenuItem("Board size")
      val exitItem  = new MenuItem("Exit")
      exitItem.accelerator = new KeyCodeCombination(KeyCode.X, KeyCombination.ControlDown)
      gameMenu.items = List(startItem, new SeparatorMenuItem, sizeItem, new SeparatorMenuItem, exitItem)
      menubar.menus = List(gameMenu)
      borderPane.top = menubar

      borderPane.center = gridPane
      
      val hbox1 = new HBox()
      val label_b = new Label{
        prefHeight = dWidth
        prefWidth = dHeight
        style = "-fx-background-image: url(" + blueflagFilepath + ");"+
            "-fx-background-size: 18px 18px;"+
            "-fx-background-repeat: no-repeat;"+
            "-fx-background-position: center;"
      }
      var text = new Text{
        fill = Blue
        text = gameState.blueFlagsCaptured.size.toString
      }
      val rectangle = new Rectangle{
        width = dWidth 
        height = dHeight
        fill = White
        effect = new InnerShadow(15, Grey)
      }
      var stack = new StackPane()
      stack.children = List(rectangle, text)
      hbox1.children = List(label_b, stack)
      val hbox2 = new HBox()
      val label_r = new Label{
        prefHeight = dWidth
        prefWidth = dHeight
        style = "-fx-background-image: url(" + redflagFilepath + ");"+
            "-fx-background-size: 18px 18px;"+
            "-fx-background-repeat: no-repeat;"+
            "-fx-background-position: center;"
      }
      var text2 = new Text{
        fill = Red
        text = gameState.redFlagsCaptured.size.toString
      }
      val rectangle2 = new Rectangle{
        width = dWidth 
        height = dHeight
        fill = White
        effect = new InnerShadow(15, Grey)
      }
      var stack2 = new StackPane()
      stack2.children = List(rectangle2, text2)
      hbox2.children = List(label_r, stack2)
      borderPane.bottom = new BorderPane{
        left = hbox1
        right = hbox2
      }
      


      borderPane.alignmentInParent = Pos.CENTER

      content = List(borderPane)

      
      var lastTime = 0L
      var timer:AnimationTimer = AnimationTimer(t => {
          if (lastTime > 0) {
            text.text = gameState.blueFlagsCaptured.size.toString
            text2.text = gameState.redFlagsCaptured.size.toString
          }
          lastTime = t
        }
      )
      timer.start

      /*
      startItem.onAction = (e: ActionEvent) => {
        val borderPaneNew = new BorderPane()
        borderPaneNew.top = menubar
        borderPaneNew.center = gridPaneNew
        content = List(borderPaneNew)
      }*/
      
      exitItem.onAction = (e: ActionEvent) => sys.exit(0)
    }
  }

  stage.resizable = false
  stage.centerOnScreen()
}
