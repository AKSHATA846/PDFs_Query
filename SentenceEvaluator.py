return re.sub(r"([a-z])([A-Z])", "\g<1> \g<2>", class_name)
return re.sub(r"([a-z])([A-Z])", r"\g<1> \g<2>", class_name)
