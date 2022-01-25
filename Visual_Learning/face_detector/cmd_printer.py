def divider(text="", char="=", line_max=60, show=True):
    if len(char) != 1:
        raise ValueError(
            "Divider chars need to be one character long. "
            "Received: {}".format(char)
        )
    deco = char * (int(round((line_max - len(text))) / 2) - 2)
    text = " {} ".format(text) if text else ""
    text = f"\n{deco}{text}{deco}"
    if len(text) < line_max:
        text = text + char * (line_max - len(text))
    if show:
        print(text)
    return text


def plain_divider(char="=", line_max=60, show=True):
    if len(char) != 1:
        raise ValueError(
            "Divider chars need to be one character long. "
            "Received: {}".format(char)
        )
    deco = char * line_max
    text = f"\n{deco}"
    if show:
        print(text)
    return text
