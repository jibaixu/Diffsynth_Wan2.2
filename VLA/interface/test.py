import numpy as np
from PIL import Image
from websocket_policy_server import ExternalRobotInferenceClient

def main():
    host = "100.64.147.46"
    port = 6666
    task_description = "Pick up the ball and place into the box"

    # 初始化推理客户端
    policy_client = ExternalRobotInferenceClient(host=host, port=port)
    policy_client.set_robot_uid('piper_real') 

    image = np.array(Image.open("example/image.png"))
    wrist_image = np.array(Image.open("example/wrist_image.png"))
    state = np.array([2.3883800e-01,  1.5919000e-02,  1.9333801e-01, -3.1350477e+00,-1.2868313e-01, -3.0819724e+00,  6.9999998e-04])
    element = {
        "video.image": np.stack([image, image], axis=0)[:, None],                         # (2,1,480,640,3) uint8
        "video.wrist_image": np.stack([wrist_image, wrist_image], axis=0)[:, None],       # (2,1,480,640,3)  uint8
        "state": np.stack([state, state], axis=0)[:, None],                               # (2,1,7) float32
        "annotation.human.task_description": [task_description,task_description],
    }
    # 推理动作
    action_chunk = policy_client.get_action(element)
    # (16, 7) action chunk大小为16
    pred_action = np.concatenate([action_chunk['action.position'],action_chunk['action.rotation'],action_chunk['action.gripper']],axis=-1)

if __name__ == "__main__":
    main()