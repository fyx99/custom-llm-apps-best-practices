from gen_ai_hub.orchestration.models.message import SystemMessage, UserMessage
from gen_ai_hub.orchestration.models.template import Template, TemplateValue
from gen_ai_hub.orchestration.models.response_format import ResponseFormatJsonSchema 
from gen_ai_hub.orchestration.models.config import OrchestrationConfig
from gen_ai_hub.orchestration.service import OrchestrationService
from gen_ai_hub.orchestration.models.llm import LLM
import json

def llm(prompt, variables, json_schema):

    response_format = ResponseFormatJsonSchema(name="schema", description="schema mapping", schema=json_schema) if json_schema else "text"

    template = Template(
        messages=[
            UserMessage(prompt)
        ],
        response_format=response_format,
        defaults=[
           
        ]
    )

    llm = LLM(name="gpt-4o", version="latest", parameters={"max_tokens": 256, "temperature": 0.2})

    config = OrchestrationConfig(
        template=template,  # or use a referenced prompt template from Step 1
        llm=llm,
    )
    
    orchestration_service = OrchestrationService(config=config)


    result = orchestration_service.run(template_values=[TemplateValue(name=key, value=value) for key, value in variables.items()])
    
    return result.orchestration_result.choices[0].message.content if not json_schema else json.loads(result.orchestration_result.choices[0].message.content)