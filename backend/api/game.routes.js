const express = require('express');
const router = express.Router();
const GameSession = require('../models/GameSession');

// Start a new game session
router.post('/session/start', async (req, res) => {
    try {
        const { playerId, missionId } = req.body;
        const session = new GameSession({ playerId, missionId });
        await session.save();
        res.status(201).json(session);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// End a game session and save results
router.post('/session/end', async (req, res) => {
    try {
        const { sessionId, success, tokensEarned } = req.body;
        const session = await GameSession.findById(sessionId);
        if (!session) {
            return res.status(404).json({ error: 'Session not found' });
        }
        session.endTime = new Date();
        session.success = success;
        session.tokensEarned = tokensEarned;
        await session.save();
        res.json(session);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

module.exports = router;
