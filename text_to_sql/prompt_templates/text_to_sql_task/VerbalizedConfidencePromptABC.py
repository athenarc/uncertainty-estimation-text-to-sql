from abc import abstractmethod
from text_to_sql.prompt_templates.text_to_sql_task.PromptABC import PromptABC


class VerbalizedConfidencePromptABC(PromptABC):

    @abstractmethod
    def get_predicted_uncertainty(self, response: str) -> int:
        """Extracts the confidence level from the model's response.

        Args:
            response (str): The model's response containing the SQL query and confidence level.

        Returns:
            float: The extracted uncertainty level as a float between 0 and 1, none if extraction fails.
        """
        pass

