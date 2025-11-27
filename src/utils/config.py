def flatten_config(config):
    """
    Convert a sweep-style config with nested parameter dicts
    into a flat key:value dict.
    """
    params = config.get("parameters", {})
    flat = {}
    for k, v in params.items():
        if "value" in v:
            flat[k] = v["value"]
        elif "values" in v:
            flat[k] = v["values"][0]
        elif "min" in v and "max" in v:
            flat[k] = v["min"]
        else:
            flat[k] = v
    return flat
