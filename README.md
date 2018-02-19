# MinesweeperFlags-AI

In this project, we train a convolutional neural network AI to play Minesweeper Flags through self-play using the Scala machine learning library deeplearning4j. Also included is a GUI written in ScalaFX allowing a human player to play against the AI.

Minesweeper flags is a two-player turn-based variant of Minesweeper, where instead of mines there are flags, and the goal is to capture flags. When a player clicks on a square revealing a flag, thereby capturing it, he/she is given another turn.

### Instructions
To run the program, download the repository, run sbt with the repository as the working directory. Finally, type 'run' in sbt.

### Examples
Here are a couple of illustrative examples where I played against the AI. Here, the red flags belong to the AI. In each example, I opened up a region (light green) with no flags. The AI then found the surrounding red flags! See if you can deduce them.


<p align="center">
  <img src="minesweeper2.png?raw=true"/>
</p>

<p align="center">
  <img src="minesweeper1.png?raw=true"/>
</p>
