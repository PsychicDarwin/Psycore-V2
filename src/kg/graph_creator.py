from abc import ABC, abstractmethod

class GraphRelation:
    def __init__(self, subject: str, object: str, relation: str):
        self.subject = subject
        self.object = object
        self.relation = relation

    def _to_dict (self):
        return {
            "subject": self.subject,
            "object": self.object,
            "relation": self.relation
        }
    
    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, GraphRelation):
            return False
        return self.subject == __value.subject and self.object == __value.object and self.relation == __value.relation
    

class GraphCreator(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def create_graph_relations(self, text: str):
        """
        Converts the text into a list of GraphRelation objects.

        :param text: The text to convert.
        :return: A list of GraphRelation objects.
        """
        return [
            GraphRelation("document", text, "CONTAINS")
        ]


    def create_graph_dict(self, text: str):
        """
        Uses the create_graph_relations method to convert the text into a graph representation.
        Then converts the graph representation into a dictionary format.
        """
        relations = self.create_graph_relations(text)
        graph_dict = {
            "triples": [relation._to_dict() for relation in relations]
        }
        return graph_dict