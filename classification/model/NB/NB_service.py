from typing import List

from bentoml import artifacts, api, BentoService
from bentoml.adapters import JsonInput
from bentoml.frameworks.sklearn import SklearnModelArtifact
from bentoml.service.artifacts.common import PickleArtifact
from bentoml.types import JsonSerializable


@artifacts([SklearnModelArtifact('model'), PickleArtifact('tokenizer'), PickleArtifact('pos_tagging')])
class category(BentoService):
    @api(input=JsonInput(), batch=True)
    def predict(self, parsed_jsons: List[JsonSerializable]):
        input_texts = parsed_jsons[0]['text']

        text = self.artifacts.pos_tagging(input_texts)
        text = self.artifacts.tokenizer.transform(text)
        pred_y = self.artifacts.model.predict(text)

        return [pred_y]