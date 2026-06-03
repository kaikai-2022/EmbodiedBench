from VLABench.tasks.dm_task import *

class PrimitiveTask(LM4ManipBaseTask):
    def reset_intention_distance(self):
        self.intention_distance = dict()
        entity_names = list(self.entities.keys())
        for ignore_entity in self.random_ignored_entities:
            if ignore_entity in entity_names:
                entity_names.remove(ignore_entity)
        for entity_name in entity_names:
            self.intention_distance[entity_name] = np.inf

    def reset_task_progress(self):
        self.target_is_grasped = dict()
        if isinstance(self.target_entity, str):
            self.target_is_grasped[self.target_entity] = False
        elif isinstance(self.target_entity, list):
            for entity in self.target_entity:
                self.target_is_grasped[entity] = False

    def update_intention_distance(self, physics):
        ee_pos = self.robot.get_end_effector_pos(physics)
        for key, entity in self.entities.items():
            if key in self.random_ignored_entities: continue
            self.intention_distance[key] = min(self.intention_distance[key], distance(ee_pos, entity.get_xpos(physics)))

    def update_task_progress(self, physics):
        if isinstance(self.target_entity, list):
            for entity in self.target_entity:
                if self.entities[entity].is_grasped(physics, self.robot):
                    self.target_is_grasped[entity] = True
        elif isinstance(self.target_entity, str):
            if self.entities[self.target_entity].is_grasped(physics, self.robot):
                self.target_is_grasped[self.target_entity] = True

    def get_intention_score(self, physics, threshold=0.2, discrete=True):
        if isinstance(self.target_entity, list):
            return self.get_intention_score_to_entity(physics, self.target_entity[-1], threshold, discrete)
        return self.get_intention_score_to_entity(physics, self.target_entity, threshold, discrete)

    def get_task_progress(self, physics):
        _, conditions_met = self.conditions.met_progress(physics)
        n_condition = len(self.conditions)
        n_condition += len(self.target_is_grasped)
        target_entity_met = []
        for value in self.target_is_grasped.values():
            if value: target_entity_met.append(value)
        return (len(conditions_met) + len(target_entity_met)) / n_condition

    def get_intention_score_to_entity(self, physics, entity_name, threshold=0.2, discrete=False):
        if discrete:
            return int(self.intention_distance[entity_name] < threshold)
        else:
            if threshold - self.intention_distance[entity_name] < 0:
                return 0
            return 1 / (1 + (threshold - self.intention_distance[entity_name]) + 1e-6)