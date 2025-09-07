import React, { useState, useEffect } from 'react';
import GameCanvas from './components/GameCanvas';
import MissionSelector from './components/MissionSelector';
import TokenDisplay from './components/TokenDisplay';
import AdManager from './components/AdManager';
import { GameProvider } from './contexts/GameContext';
import './App.css';

function App() {
  const [gameState, setGameState] = useState('menu');
  const [tokens, setTokens] = useState(0);
  const [playerData, setPlayerData] = useState(null);

  useEffect(() => {
    // Load player data
    const loadPlayerData = async () => {
      const data = await localStorage.getItem('ghosts_player_data');
      if (data) {
        setPlayerData(JSON.parse(data));
        setTokens(JSON.parse(data).tokens || 0);
      }
    };
    
    loadPlayerData();
  }, []);

  const startMission = (missionId) => {
    setGameState('playing');
    // Initialize mission
  };

  const completeMission = (success, earnedTokens) => {
    setGameState('menu');
    if (success) {
      setTokens(prev => prev + earnedTokens);
      // Update player data
    }
  };

  const showAd = () => {
    // Show ad and handle reward
    return new Promise((resolve) => {
      // Ad implementation
      resolve(true);
    });
  };

  return (
    <GameProvider value={{ tokens, setTokens, playerData, showAd }}>
      <div className="App">
        {gameState === 'menu' && (
          <div className="menu-container">
            <TokenDisplay />
            <MissionSelector onSelectMission={startMission} />
          </div>
        )}
        {gameState === 'playing' && (
          <GameCanvas onMissionComplete={completeMission} />
        )}
        <AdManager />
      </div>
    </GameProvider>
  );
}

export default App;
