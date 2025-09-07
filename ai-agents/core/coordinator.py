import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Any
from .character_agent import CharacterCreationAgent
from .environment_agent import EnvironmentAgent
from .voice_agent import VoiceAgent
from .weapon_agent import WeaponAgent
from .deployment_agent import DeploymentAgent

class AICoordinator:
    def __init__(self):
        self.agents = {
            'character': CharacterCreationAgent(),
            'environment': EnvironmentAgent(),
            'voice': VoiceAgent(),
            'weapon': WeaponAgent(),
            'deployment': DeploymentAgent()
        }
        self.task_queue = asyncio.Queue()
        self.active_tasks = {}
        self.logger = self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('ai_coordinator.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger('AICoordinator')
    
    async def process_tasks(self):
        """Continuously process tasks from the queue"""
        while True:
            task = await self.task_queue.get()
            try:
                self.logger.info(f"Processing task: {task['type']}")
                result = await self.execute_task(task)
                self.logger.info(f"Task completed: {task['type']}")
                
                # Store result and trigger next actions
                await self.handle_task_result(task, result)
                
            except Exception as e:
                self.logger.error(f"Task failed: {task['type']} - {str(e)}")
            finally:
                self.task_queue.task_done()
    
    async def execute_task(self, task: Dict[str, Any]):
        """Execute a task with the appropriate agent"""
        task_type = task['type']
        
        if task_type == 'create_character':
            return await self.agents['character'].create_character(task['params'])
        elif task_type == 'generate_environment':
            return await self.agents['environment'].generate_environment(task['params'])
        elif task_type == 'generate_voice_lines':
            return await self.agents['voice'].generate_dialogue(task['params'])
        elif task_type == 'design_weapon':
            return await self.agents['weapon'].design_weapon(task['params'])
        elif task_type == 'deploy_build':
            return await self.agents['deployment'].deploy(task['params'])
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    async def handle_task_result(self, task: Dict[str, Any], result: Any):
        """Handle task results and trigger subsequent actions"""
        # Store result in database
        # Trigger dependent tasks
        # Update UI/dashboard
        
        if task['type'] == 'create_character' and result['success']:
            # Character created, now generate voice lines
            voice_task = {
                'type': 'generate_voice_lines',
                'params': {
                    'character_id': result['character_id'],
                    'personality': result['personality_traits']
                }
            }
            await self.add_task(voice_task)
    
    async def add_task(self, task: Dict[str, Any]):
        """Add a new task to the queue"""
        task['id'] = self.generate_task_id()
        task['created_at'] = datetime.now().isoformat()
        task['status'] = 'queued'
        
        await self.task_queue.put(task)
        self.active_tasks[task['id']] = task
        
        self.logger.info(f"Task added to queue: {task['type']}")
        return task['id']
    
    def generate_task_id(self):
        """Generate a unique task ID"""
        return f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(datetime.now()))}"
    
    def get_agent_status(self):
        """Get status of all agents"""
        return {
            agent_name: {
                'status': agent.get_status(),
                'capabilities': agent.get_capabilities(),
                'current_task': agent.current_task
            }
            for agent_name, agent in self.agents.items()
        }
    
    async def shutdown(self):
        """Gracefully shutdown all agents"""
        self.logger.info("Shutting down AI coordinator...")
        for agent in self.agents.values():
            await agent.shutdown()
        self.logger.info("AI coordinator shutdown complete")

# Global coordinator instance
coordinator = AICoordinator()
