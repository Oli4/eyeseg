def find_volumes(data_path):
    if data_path.is_dir():
        vol_volumes = data_path.glob("**/*.vol")
        xml_volumes = data_path.glob("**/*.xml")
        eye_volumes = data_path.glob("**/*.eye")
        # We do not support multiple XML exports in the same folder.
        xml_volumes = [v.parent for v in xml_volumes]
    elif data_path.is_file():
        if ".vol" == data_path.suffix:
            vol_volumes = [data_path]
            xml_volumes = []
            eye_volumes = []
        if ".xml" == data_path.suffix:
            xml_volumes = [data_path]
            vol_volumes = []
            eye_volumes = []
        if ".eye" == data_path.suffix:
            eye_volumes = [data_path]
            xml_volumes = []
            vol_volumes = []
    else:
        raise ValueError("Data not found")

    return {"vol": set(vol_volumes), "xml": set(xml_volumes), "eye": set(eye_volumes)}
