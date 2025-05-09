from abc import ABC, abstractmethod
from system_manager.LoggerController import LoggerController

# Configure logging
logger = LoggerController.get_logger()

class GraphRelation:
    def __init__(self, subject: str, object: str, relation: str):
        logger.debug(f"Creating new GraphRelation: subject='{subject}', object='{object}', relation='{relation}'")
        self.subject = subject
        self.object = object
        self.relation = relation

    def _to_dict(self):
        logger.debug(f"Converting GraphRelation to dict: {self.subject} -> {self.relation} -> {self.object}")
        return {
            "subject": self.subject,
            "object": self.object,
            "relation": self.relation
        }
    
    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, GraphRelation):
            logger.debug(f"Comparison failed: {__value} is not a GraphRelation")
            return False
        is_equal = self.subject == __value.subject and self.object == __value.object and self.relation == __value.relation
        logger.debug(f"Comparing GraphRelations: {self.subject}->{self.relation}->{self.object} == {__value.subject}->{__value.relation}->{__value.object} = {is_equal}")
        return is_equal

    def __dict_to_relation(self, data: dict):
        logger.debug(f"Converting dict to GraphRelation: {data}")
        if not isinstance(data, dict):
            logger.error(f"Invalid data type: {type(data)}")
            raise TypeError("data must be a dictionary")
        if "subject" not in data or "object" not in data or "relation" not in data:
            logger.error(f"Missing required keys in data: {data}")
            raise ValueError("data must contain 'subject', 'object', and 'relation' keys")
        return GraphRelation(data["subject"], data["object"], data["relation"])

class GraphCreator(ABC):
    @abstractmethod
    def __init__(self):
        logger.debug("Initializing GraphCreator")
        pass

    @abstractmethod
    def create_graph_relations(self, text: str):
        """
        Converts the text into a list of GraphRelation objects.

        :param text: The text to convert.
        :return: A list of GraphRelation objects.
        """
        logger.debug(f"Creating default graph relation for text of length {len(text)}")
        return [
            GraphRelation("document", text, "CONTAINS")
        ]

    def create_graph_dict(self, text: str):
        """
        Uses the create_graph_relations method to convert the text into a graph representation.
        Then converts the graph representation into a dictionary format.
        """
        logger.debug(f"Starting graph creation for text of length {len(text)}")
        relations = self.create_graph_relations(text)
        logger.debug(f"Created {len(relations)} relations")
        
        graph_dict = {
            "triples": [relation._to_dict() for relation in relations]
        }
        logger.debug(f"Converted {len(relations)} relations to dictionary format")
        return graph_dict