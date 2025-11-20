from collections import defaultdict, deque
import heapq

class RelationGraph:
    def __init__(self):
        self.graph = defaultdict(lambda: defaultdict(dict))  # forward
        self.reverse_index = defaultdict(lambda: defaultdict(dict))  # backward
        self._pending_nodes = set()
        # status tracking
        self._processed_levels = defaultdict(lambda: defaultdict(int))

    def add_relation(
        self,
        source_topic: str,
        source_item: str,
        relations: dict,
        overwrite: bool = False,
    ):
        for target_topic, target_item in relations.items():
            if source_topic == target_topic:
                continue

            if overwrite:
                self.graph[source_topic][source_item][target_topic] = target_item
                self.reverse_index[target_topic][target_item][
                    source_topic
                ] = source_item
                self._pending_nodes.add((target_topic, target_item))
            else:
                existing = self.graph[source_topic][source_item].get(target_topic)
                if existing and existing != target_item:
                    raise ValueError(
                        f"关系冲突: {source_topic}.{source_item} -> {target_topic}({target_item}) 已有映射 {existing}"
                    )
                self.graph[source_topic][source_item][target_topic] = target_item

                # Update reverse index
                rev_conflict = self.reverse_index[target_topic][target_item].get(
                    source_topic
                )
                if rev_conflict is None:
                    self.reverse_index[target_topic][target_item][
                        source_topic
                    ] = source_item
                    self._pending_nodes.add((target_topic, target_item))

        self._pending_nodes.add((source_topic, source_item))
        for t_topic, t_item in relations.items():
            self._pending_nodes.add((t_topic, t_item))

    def get_relations(self, topic: str, item: str, max_depth=100) -> dict:

        self.deduce_relations(max_iterations=max_depth)

        results = {}
        visited = set()
        queue = deque([(topic, item, 0)])

        while queue:
            current_topic, current_item, depth = queue.popleft()
            if depth > max_depth:
                continue
            if (current_topic, current_item) in visited:
                continue
            visited.add((current_topic, current_item))

            for rel_topic, rel_item in sorted(
                self.graph[current_topic].get(current_item, {}).items()
            ):
                if rel_topic != topic and rel_topic not in results:
                    results[rel_topic] = rel_item
                    queue.append((rel_topic, rel_item, depth + 1))

            for src_topic, src_item in sorted(
                self.reverse_index[current_topic].get(current_item, {}).items()
            ):
                if src_topic != topic and src_topic not in results:
                    results[src_topic] = src_item
                    queue.append((src_topic, src_item, depth + 1))

        return results

    def deduce_relations(self, max_iterations=100):
        priority_queue = []

        sorted_nodes = sorted(
            self._pending_nodes,
            key=lambda x: (self._processed_levels[x[0]][x[1]], x[0], x[1]),
        )

        for topic, item in sorted_nodes:
            next_level = self._processed_levels[topic][item] + 1
            heapq.heappush(priority_queue, (next_level, topic, item))

        self._pending_nodes.clear()
        processed_count = 0

        while priority_queue and processed_count < max_iterations:
            level, current_topic, current_item = heapq.heappop(priority_queue)
            if self._processed_levels[current_topic][current_item] >= level:
                continue
            self._process_node(current_topic, current_item, level, priority_queue)
            processed_count += 1

    def _process_node(self, topic: str, item: str, level: int, priority_queue: list):
        current_rels = self.graph[topic][item]

        for mid_topic, mid_item in sorted(current_rels.items()):
            for dst_topic, dst_item in sorted(
                self.graph[mid_topic].get(mid_item, {}).items()
            ):
                if dst_topic == topic:
                    continue
                if dst_topic not in current_rels:
                    self._add_relation_safe(topic, item, dst_topic, dst_item)
                    self._schedule_update(topic, item, level + 1, priority_queue)

        for src_topic, src_item in sorted(
            self.reverse_index[topic].get(item, {}).items()
        ):
            for fwd_topic, fwd_item in sorted(
                self.graph[src_topic].get(src_item, {}).items()
            ):
                if fwd_topic != topic and fwd_topic not in current_rels:
                    self._add_relation_safe(topic, item, fwd_topic, fwd_item)
                    self._schedule_update(topic, item, level + 1, priority_queue)

            for rev_topic, rev_item in sorted(
                self.reverse_index[src_topic].get(src_item, {}).items()
            ):
                if rev_topic != topic and rev_topic not in current_rels:
                    self._add_relation_safe(topic, item, rev_topic, rev_item)
                    self._schedule_update(topic, item, level + 1, priority_queue)

        self._processed_levels[topic][item] = level

    def _add_relation_safe(
        self, src_topic: str, src_item: str, dst_topic: str, dst_item: str
    ):
        if dst_topic not in self.graph[src_topic][src_item]:
            self.graph[src_topic][src_item][dst_topic] = dst_item
            self.reverse_index[dst_topic][dst_item][src_topic] = src_item
            self._pending_nodes.add((src_topic, src_item))
            self._pending_nodes.add((dst_topic, dst_item))

    def _schedule_update(self, topic: str, item: str, level: int, priority_queue: list):
        if self._processed_levels[topic][item] < level:
            heapq.heappush(priority_queue, (level, topic, item))
            self._processed_levels[topic][item] = level


if __name__ == "__main__":
    graph = RelationGraph()

    graph.add_relation("C", "c2", {"A": "a3"})
    graph.add_relation("C", "c3", {"A": "a6"})
    graph.add_relation("C", "c4", {"A": "a8"})
    graph.add_relation("D", "d1", {"C": "c1"})

    graph.add_relation("A", "a1", {"B": "b1"})
    graph.add_relation("A", "a3", {"B": "b2"})
    graph.add_relation("C", "c1", {"A": "a1"})

    graph.deduce_relations()

    print(graph.get_relations("B", "b1"))
    print(graph.get_relations("B", "b2"))
    print(graph.get_relations("C", "c3"))
