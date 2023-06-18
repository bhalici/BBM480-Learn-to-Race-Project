![Learn to Race Banner](docs/l2r_banner.jpg)

# BBM479/480 



#  Getting Started
1. **Sign up** to join the competition [on the AIcrowd website](https://www.aicrowd.com/challenges/learn-to-race-autonomous-racing-virtual-challenge).
2. **Download** the Arrival Autonomous Racing Simulator [from this link](https://www.aicrowd.com/clef_tasks/82/task_dataset_files?challenge_id=954).
3. **Fork** this starter kit repository. You can use [this link](https://gitlab.aicrowd.com/learn-to-race/l2r-starter-kit/-/forks/new) to create a fork.
4. **Clone** your forked repo and start developing your autonomous racing agent.
5. **Develop** your autonomous racing agents following the template in [how to write your own agent](#how-to-write-your-own-agent) section.
6. [**Submit**](#how-to-make-a-submission) your trained models to [AIcrowd Gitlab](https://gitlab.aicrowd.com) for evaluation [(full instructions below)](#how-to-make-a-submission). The automated evaluation setup will evaluate the submissions on the racetrack and report the metrics on the leaderboard of the competition.

# How to start participating?

## Setup

1. **Fork the repository**. You can use [this link](https://gitlab.aicrowd.com/learn-to-race/l2r-starter-kit/-/forks/new) to create a fork.

2.  **Clone the repository**

    ```
    git clone git@gitlab.aicrowd.com:<YOUR_AICROWD_USERNAME>/l2r-starter-kit.git
    ```

3. **Install** competition specific dependencies!
    ```
    cd l2r-starter-kit
    pip install -r requirements.txt
    ```

4. Try out the SAC agent by running `python rollout.py`. You should start the simulator first, by running `bash <simulator_path>/ArrivalSim-linux-0.7.1.188691/LinuxNoEditor/ArrivalSim.sh -openGL`. You can also checkout the [random agent](agents/random_agent.py) implementation for a minimal reference code.

5. Write your own agent as described in [How to write your own agent](#how-to-write-your-own-agent) section.

6. Make a submission as described in [How to make a submission](#how-to-make-a-submission) section.






## Contributors 

- İlker Emre KOÇ 21827608  
- Selçuk YILMAZ  21828035  
- Buğrahan HALICI 21827483
- İrem Atak  21726933  

