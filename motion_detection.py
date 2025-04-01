import numpy as np

def analyze_and_log_context(previous_positions, current_positions, logged_ids, object_id, label, conf, log_function):
    """Determines context based on movement, logs the detection, and returns the context."""

    # Default context is None; it will be updated if motion or waving is detected
    context = None

    # 1. Check for motion based on position change (even for the same object ID)
    if object_id in previous_positions and object_id in current_positions:
        prev_pos = np.array(previous_positions[object_id])
        curr_pos = np.array(current_positions[object_id])
        movement = np.linalg.norm(curr_pos - prev_pos)
        
        if movement < 10:
            context = "Object is stationary"
        elif movement < 30:
            context = "Object is moving slowly"
        else:
            context = "Object is moving fast"
    elif object_id not in previous_positions:
        context = "Object detected for the first time"

    # 2. Log the context if determined (motion or gesture)
    if context:
        # Check if the object was already logged for this context (to avoid duplicate logs)
        if object_id not in logged_ids or logged_ids[object_id] != context:
            log_function(logged_ids, object_id, label, conf, context)
            logged_ids[object_id] = context  # Update the logged context

    return context  # Return context for debugging or other uses