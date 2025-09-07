import asyncio
from typing import Dict, List, Optional
from dataclasses import dataclass
from .coordinator import AITask
import heapq

@dataclass
class PrioritizedTask:
    priority: int
    task: AITask

class TaskQueue:
    def __init__(self):
        self.queue = []
        self.task_map = {}  # task_id -> PrioritizedTask
        self.lock = asyncio.Lock()
    
    async def add_task(self, task: AITask, priority: int = 1):
        """Add a task to the queue with a priority"""
        async with self.lock:
            prioritized_task = PrioritizedTask(priority, task)
            heapq.heappush(self.queue, (-priority, task.id))  # Negative for max heap
            self.task_map[task.id] = prioritized_task
    
    async def get_next_task(self) -> Optional[AITask]:
        """Get the next task from the queue based on priority"""
        async with self.lock:
            if not self.queue:
                return None
            
            # Get the highest priority task
            priority, task_id = heapq.heappop(self.queue)
            prioritized_task = self.task_map.get(task_id)
            
            if prioritized_task:
                del self.task_map[task_id]
                return prioritized_task.task
            
            return None
    
    async def remove_task(self, task_id: str) -> bool:
        """Remove a task from the queue"""
        async with self.lock:
            if task_id in self.task_map:
                del self.task_map[task_id]
                
                # Rebuild the heap without the removed task
                new_queue = []
                for priority, t_id in self.queue:
                    if t_id != task_id:
                        new_queue.append((priority, t_id))
                
                heapq.heapify(new_queue)
                self.queue = new_queue
                
                return True
            
            return False
    
    async def get_queue_size(self) -> int:
        """Get the current size of the queue"""
        async with self.lock:
            return len(self.queue)
    
    async def get_queue_state(self) -> List[Dict[str, Any]]:
        """Get the current state of the queue"""
        async with self.lock:
            state = []
            for priority, task_id in self.queue:
                prioritized_task = self.task_map.get(task_id)
                if prioritized_task:
                    state.append({
                        'task_id': task_id,
                        'type': prioritized_task.task.type,
                        'priority': prioritized_task.priority,
                        'created_at': prioritized_task.task.created_at.isoformat()
                    })
            return state
