def find_volumes(data_path):
    if data_path.is_dir():
        vol_volumes = data_path.glob("**/*.vol")
        xml_volumes = data_path.glob("**/*.xml")
        # We do not support multiple XML exports in the same folder.
        xml_volumes = [v.parent for v in xml_volumes]
    elif data_path.is_file():
        if ".vol" == data_path.suffix:
            vol_volumes = [data_path]
            xml_volumes = []
        if ".xml" == data_path.suffix:
            xml_volumes = [data_path]
            vol_volumes = []
    else:
        raise ValueError("Data not found")

    return {"vol": set(vol_volumes), "xml": set(xml_volumes)}
