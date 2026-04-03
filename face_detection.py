def detect_face(image):
    """
    Detect and extract face from an image.
    """
    if isinstance(image, Image.Image):
        # Convert PIL Image to OpenCV format
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    height, width = image.shape[:2]
    
    # Create a blob from the image
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
    
    # Set the blob as input and run the network
    net.setInput(blob)
    detections = net.forward()
    
    # Find face with highest confidence
    max_confidence = 0
    max_detection = None
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > max_confidence:
            max_confidence = confidence
            max_detection = detections[0, 0, i]
    
    # If no face detected or confidence too low
    if max_confidence < 0.1:
        return None
    
    # Get coordinates
    box = max_detection[3:7] * np.array([width, height, width, height])
    (startX, startY, endX, endY) = box.astype("int")
    
    # Extract face ROI
    face = image[startY:endY, startX:endX]
    if face.size == 0:
        return None
    
    # Convert back to RGB for further processing
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    
    # Apply color correction and normalization
    face = cv2.normalize(face, None, 0, 255, cv2.NORM_MINMAX)
    
    return face 