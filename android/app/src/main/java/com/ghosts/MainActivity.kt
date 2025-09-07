package com.ghosts

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import com.google.android.gms.ads.AdRequest
import com.google.android.gms.ads.MobileAds

class MainActivity : AppCompatActivity() {
    private lateinit var gameView: GameView
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Initialize AdMob
        MobileAds.initialize(this) {}
        
        // Initialize game view
        gameView = GameView(this)
        setContentView(gameView)
        
        // Load player data
        loadPlayerData()
    }
    
    override fun onPause() {
        super.onPause()
        gameView.pause()
    }
    
    override fun onResume() {
        super.onResume()
        gameView.resume()
    }
    
    private fun loadPlayerData() {
        // Load player data from local storage or backend
    }
    
    fun showAd(callback: (Boolean) -> Unit) {
        // Show rewarded ad
        val adRequest = AdRequest.Builder().build()
        // Ad implementation here
    }
}
