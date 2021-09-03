

package xikai.minesweeper

import scala.util.Random

case class EpGreedyPlayer(qnn: QNNPlayer, ep: Double = 0.1) {
  val random = new Random()

  def bestMove(gs: GameState): Option[Move] = {
    if (gs.isGameOver) None
    else if (Random.nextDouble() < ep)
      RandomMovePlayer.bestMove(gs)
    else qnn.bestMove(gs)
  }

}

object EpGreedyPlayer {
  def toPlayer(player: EpGreedyPlayer): Player[EpGreedyPlayer] = new Player[EpGreedyPlayer] {
    def bestMove(gs: GameState): Option[Move] = player.bestMove(gs)
  }
}
