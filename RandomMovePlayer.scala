
package xikai.minesweeper

import scala.util.Random

case class RandomMovePlayer()

object RandomMovePlayer {
  val random = new Random()

  def bestMove(gs: GameState): Option[Move] = {
    if (gs.isGameOver) None
    else {
      val validMoves =
        for (x <- 0 until gs.boardSize;
             y <- 0 until gs.boardSize;
          if ! gs.visible(x)(y))
        yield (x, y)

      if (validMoves.length == 0) {
        None
      } else {
        val i = random.nextInt(validMoves.length)
        Some(validMoves(i))
      }
    }
  }
}
