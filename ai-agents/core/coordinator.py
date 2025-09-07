import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum
import aiohttp
from dataclasses import dataclass
from .task_queue import TaskQueue
from .monitoring import PerformanceMonitor

class TaskStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class AITask:
    id: str
    type: str
    params: Dict[str, Any]
    status: TaskStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class AICoordinator:
    def __init__(self):
        self.agents = {}
        self.task_queue = TaskQueue()
        self.active_tasks = {}
        self.completed_tasks = {}
        self.performance_monitor = PerformanceMonitor()
        self.logger = self.setup_logging()
        self.session = None
        
    def setup_logging(self):
        """Setup logging for the coordinator"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/ai_coordinator.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger('AICoordinator')
    
    def register_agent(self, agent_name: str, agent_instance):
        """Register an AI agent with the coordinator"""
        self.agents[agent_name] = agent_instance
        self.logger.info(f"Registered agent: {agent_name}")
    
    async def initialize(self):
        """Initialize the coordinator and all agents"""
        self.logger.info("Initializing AI Coordinator...")
        self.session = aiohttp.ClientSession()
        
        # Initialize all registered agents
        for agent_name, agent in self.agents.items():
            if hasattr(agent, 'initialize'):
                await agent.initialize()
            self.logger.info(f"Initialized agent: {agent_name}")
        
        # Start task processing loop
        asyncio.create_task(self.process_tasks())
        self.logger.info("AI Coordinator initialized successfully")
    
    async def add_task(self, task_type: str, params: Dict[str, Any], priority: int = 1) -> str:
        """Add a new task to the queue"""
        task_id = self.generate_task_id(task_type)
        task = AITask(
            id=task_id,
            type=task_type,
            params=params,
            status=TaskStatus.PENDING,
            created_at=datetime.now()
        )
        
        await self.task_queue.add_task(task, priority)
        self.active_tasks[task_id] = task
        
        self.logger.info(f"Added task to queue: {task_type} (ID: {task_id})")
        return task_id
    
    async def process_tasks(self):
        """Continuously process tasks from the queue"""
        self.logger.info("Starting task processing loop...")
        
        while True:
            try:
                # Get next task from queue
                task = await self.task_queue.get_next_task()
                if not task:
                    await asyncio.sleep(0.1)
                    continue
                
                # Update task status
                task.status = TaskStatus.PROCESSING
                task.started_at = datetime.now()
                
                self.logger.info(f"Processing task: {task.type} (ID: {task.id})")
                
                # Execute the task with the appropriate agent
                result = await self.execute_task(task)
                
                # Update task with result
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now()
                task.result = result
                
                # Move to completed tasks
                self.completed_tasks[task.id] = task
                if task.id in self.active_tasks:
                    del self.active_tasks[task.id]
                
                self.logger.info(f"Completed task: {task.type} (ID: {task.id})")
                
                # Record performance metrics
                processing_time = (task.completed_at - task.started_at).total_seconds()
                self.performance_monitor.record_task_metrics(
                    task.type, processing_time, True
                )
                
                # Handle task result and trigger dependent tasks
                await self.handle_task_result(task, result)
                
            except Exception as e:
                self.logger.error(f"Error processing task: {str(e)}")
                if task:
                    task.status = TaskStatus.FAILED
                    task.error = str(e)
                    
                    # Record failure metrics
                    self.performance_monitor.record_task_metrics(
                        task.type, 0, False
                    )
    
    async def execute_task(self, task: AITask) -> Dict[str, Any]:
        """Execute a task with the appropriate agent"""
        task_type = task.type
        
        if task_type.startswith('character_'):
            agent_name = 'character'
            method_name = task_type.replace('character_', '')
        elif task_type.startswith('environment_'):
            agent_name = 'environment'
            method_name = task_type.replace('environment_', '')
        elif task_type.startswith('voice_'):
            agent_name = 'voice'
            method_name = task_type.replace('voice_', '')
        elif task_type.startswith('weapon_'):
            agent_name = 'weapon'
            method_name = task_type.replace('weapon_', '')
        elif task_type.startswith('deployment_'):
            agent_name = 'deployment'
            method_name = task_type.replace('deployment_', '')
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
        if agent_name not in self.agents:
            raise ValueError(f"Agent not registered: {agent_name}")
        
        agent = self.agents[agent_name]
        
        if not hasattr(agent, method_name):
            raise ValueError(f"Method not found: {method_name} on agent {agent_name}")
        
        method = getattr(agent, method_name)
        
        # Execute the method
        if asyncio.iscoroutinefunction(method):
            result = await method(task.params)
        else:
            result = method(task.params)
        
        return result
    
    async def handle_task_result(self, task: AITask, result: Dict[str, Any]):
        """Handle task results and trigger subsequent actions"""
        # This method would handle task dependencies and trigger new tasks
        # based on the results of completed tasks
        
        if task.type == 'character_create' and result.get('success'):
            # Character created, now generate voice lines
            voice_task_id = await self.add_task(
                'voice_generate_dialogue',
                {
                    'character_id': result['character_id'],
                    'personality': result.get('personality_traits', {})
                }
            )
            self.logger.info(f"Triggered voice generation task: {voice_task_id}")
        
        elif task.type == 'environment_generate' and result.get('success'):
            # Environment created, now populate with props
            props_task_id = await self.add_task(
                'environment_populate_props',
                {
                    'environment_id': result['environment_id'],
                    'environment_type': result['environment_type']
                }
            )
            self.logger.info(f"Triggered environment population task: {props_task_id}")
    
    def generate_task_id(self, task_type: str) -> str:
        """Generate a unique task ID"""
        timestamp = int(time.time() * 1000)
        random_suffix = hash(str(datetime.now())) % 10000
        return f"{task_type}_{timestamp}_{random_suffix}"
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        status = {}
        for agent_name, agent in self.agents.items():
            status[agent_name] = {
                'status': agent.get_status() if hasattr(agent, 'get_status') else 'unknown',
                'capabilities': agent.get_capabilities() if hasattr(agent, 'get_capabilities') else [],
                'current_task': agent.current_task if hasattr(agent, 'current_task') else None,
                'performance': self.performance_monitor.get_agent_metrics(agent_name)
            }
        return status
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task"""
        task = self.active_tasks.get(task_id) or self.completed_tasks.get(task_id)
        if not task:
            return None
        
        return {
            'id': task.id,
            'type': task.type,
            'status': task.status.value,
            'created_at': task.created_at.isoformat(),
            'started_at': task.started_at.isoformat() if task.started_at else None,
            'completed_at': task.completed_at.isoformat() if task.completed_at else None,
            'result': task.result,
            'error': task.error
        }
    
    async def shutdown(self):
        """Gracefully shutdown all agents"""
        self.logger.info("Shutting down AI coordinator...")
        
        # Close HTTP session
        if self.session:
            await self.session.close()
        
        # Shutdown all agents
        for agent_name, agent in self.agents.items():
            if hasattr(agent, 'shutdown'):
                if asyncio.iscoroutinefunction(agent.shutdown):
                    await agent.shutdown()
                else:
                    agent.shutdown()
            self.logger.info(f"Shutdown agent: {agent_name}")
        
        self.logger.info("AI coordinator shutdown complete")

# Global coordinator instance
coordinator = AICoordinator()
