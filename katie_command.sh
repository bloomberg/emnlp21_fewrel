# Make sure these patterns match your credentials
IDENTS="${USER}-bcs ${USER}-dob2-gen spectro-oauth-${USER}"
BRANCH=`git branch --show-current`

KATIE_BASE_CMD="katie compute run distributed-pytorch \
  --compute-framework pytorch-1.6-python-3.7 \
  --identities ${IDENTS} \
  --num-workers 1 \
  --node-size Custom \
  --node-num-cores 32 \
  --node-memory 64Gi \
  --node-num-gpus 1 \
  --pip-packages torch==1.6 transformers sklearn git+https://bbgithub.dev.bloomberg.com/sbrody18/FewRel@${BRANCH}
  --python-module fewrel.train_demo \
  --sync-launch tail \
  -- \
"
  
${KATIE_BASE_CMD} $@
