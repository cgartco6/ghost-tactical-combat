package com.ghosts

import android.content.Context
import android.graphics.Canvas
import android.view.SurfaceHolder
import android.view.SurfaceView

class GameEngine(context: Context) : SurfaceView(context), SurfaceHolder.Callback, Runnable {
    private var gameThread: Thread? = null
    private var isRunning = false
    private var canvas: Canvas? = null
    private val holder: SurfaceHolder = holder

    init {
        holder.addCallback(this)
    }

    override fun surfaceCreated(holder: SurfaceHolder) {
        startGame()
    }

    override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {
        // Handle surface changes
    }

    override fun surfaceDestroyed(holder: SurfaceHolder) {
        stopGame()
    }

    private fun startGame() {
        isRunning = true
        gameThread = Thread(this)
        gameThread?.start()
    }

    private fun stopGame() {
        isRunning = false
        gameThread?.join()
    }

    override fun run() {
        while (isRunning) {
            update()
            render()
        }
    }

    private fun update() {
        // Update game state
    }

    private fun render() {
        canvas = holder.lockCanvas()
        canvas?.let {
            // Draw game elements
            holder.unlockCanvasAndPost(it)
        }
    }
}
