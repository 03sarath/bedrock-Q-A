import json
import boto3

def lambda_handler(event, context):
    bedrock = boto3.client('bedrock-runtime')
    
    if event.get('httpMethod') == 'OPTIONS':
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Methods': 'POST,OPTIONS'
            },
            'body': ''
        }
    
    try:
        body = json.loads(event.get('body', '{}'))
        pdf_text = body.get('context', '')
        question = body.get('question', '')
        
        if not pdf_text or not question:
            return {
                'statusCode': 400,
                'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Content-Type': 'application/json'
                },
                'body': json.dumps({'error': 'Missing context or question'})
            }
        
        prompt_data = f"""Answer the question based only on the information provided between ##.
Only answer the question if you can find relevant information in the context, otherwise, answer "I don't have enough information to answer this question".
#
{pdf_text}
#
Question: {question}
Answer:"""
        
        bedrock_body = json.dumps({
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": prompt_data}]
                }
            ],
            "inferenceConfig": {
                "maxTokens": 512,
                "temperature": 0,
                "topP": 0.9
            }
        })
        
        response = bedrock.invoke_model(
            body=bedrock_body,
            modelId="us.amazon.nova-2-lite-v1:0",  # Nova 2 Lite
            accept="application/json",
            contentType="application/json"
        )
        
        response_body = json.loads(response.get("body").read())
        answer = response_body["output"]["message"]["content"][0]["text"]
        
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'application/json'
            },
            'body': json.dumps({'answer': answer.strip()})
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'application/json'
            },
            'body': json.dumps({'error': str(e)})
        }
