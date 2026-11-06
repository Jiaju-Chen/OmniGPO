"""CPU-only numerical example. No model weights, dataset, Ray, or environment."""
import numpy as np
import torch

from omnigpo import Batch, StandardConfig, compute_credit, critic_loss, compute_policy_loss


def tiny_batch():
    mask=torch.tensor([[1.,1.,0.],[1.,1.,1.],[1.,0.,0.],
                       [1.,1.,1.],[1.,0.,0.],[1.,1.,0.]])
    rewards=np.array([0.,0.,10.,0.,0.,0.])
    outcome=torch.zeros_like(mask)
    for i in range(6):outcome[i,int(mask[i].sum())-1]=10. if i<3 else 0.
    outcome[1,2]-=.1
    return Batch(batch=dict(response_mask=mask,token_level_rewards=outcome,
        values=torch.arange(18,dtype=torch.float32).reshape(6,3)/20),
        non_tensor_batch=dict(uid=np.array(['task']*6),traj_uid=np.array(['A']*3+['B']*3),
        step_idx=np.array([0,1,2,0,1,2]),anchor_obs=np.array(['s','a','c','s','b','c']),
        rewards=rewards,terminal_kind=np.array(['','','success','','','failure'])))


def main():
    cfg=StandardConfig()
    batch=compute_credit(tiny_batch(),cfg)
    new_values=batch.batch['values'].detach().clone().requires_grad_()
    loss,metrics=critic_loss(new_values,batch,cfg)
    loss.backward()
    # Token ratios/clipping, not a product across a whole action response.
    old_logp=torch.full_like(new_values,-1.)
    new_logp=(old_logp+.02).requires_grad_()
    actor_loss,*_=compute_policy_loss(old_logp,new_logp,batch.batch['advantages'],
        batch.batch['response_mask'],cliprange=.2)
    actor_loss.backward()
    print('q=',cfg.q,'beta=',cfg.beta)
    print('actor_credit=',batch.batch['advantages'][:,0].tolist())
    print('critic_loss=',float(loss),'actor_loss=',float(actor_loss))
    print('critic_grad_finite=',bool(torch.isfinite(new_values.grad).all()))
    print('graph_teacher_metrics=',batch.meta_info['graph_teacher_metrics'])


if __name__=='__main__':main()
