Adapt dopamine to control problem

C:\ProgramData\Anaconda3\Scripts\activate base

python -um dopamine.ctrl.train --agent_name=dqn --base_dir=tmp/dop --gin_files=dopamine/agents/dqn/configs/dqn.gin

python -um dopamine.ctrl.train --agent_name=rainbow --base_dir=tmp/dop --gin_files=dopamine/agents/rainbow/configs/c51.gin