def route(message):
    text = message.lower()
    if 'sign in' in text or 'login' in text:
        return 'account support'
    return 'general support'

print(route('I cannot sign in'))
