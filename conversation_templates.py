from prompts import (
    PPP_LLAVA_SYSTEM_PROMPT,
    PPP_LLAVA_USER_PROMPT,
    PPP_GPT_SYSTEM_PROMPT,
    PPP_GPT_USER_PROMPT
)

PPP_LLAVA_CONVERSATION_TEMPLATE = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": PPP_LLAVA_SYSTEM_PROMPT}
        ]
    },
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": PPP_LLAVA_USER_PROMPT}
        ]
    }
]

PPP_GPT_SYSTEM_CONVERSATION_TEMPLATE = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": PPP_GPT_SYSTEM_PROMPT}
        ]
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": PPP_GPT_USER_PROMPT},
            {
                "type": "image_url",
                "image_url": {"url": "<BASE64_IMAGE_DATA>"}
            }
        ]
    }
]
