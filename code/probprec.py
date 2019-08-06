# Playground version!


import torch
import numpy as np
from scipy.sparse import linalg as splinalg
dtype = torch.float
# torch.set_default_tensor_type(torch.float)

# The Preconditioner class runs as an Optimizer.


class Preconditioner(torch.optim.Optimizer):

    def __init__(self, params, est_rank=2, num_observations=5, prior_iterations=10, weight_decay=0, lr=None,
                 optim_class=torch.optim.SGD, optim_hyperparams={}):
        if not 0 <= est_rank:
            raise ValueError("Invalid Hessian rank: {}".format(est_rank))
        if not 0 <= num_observations:
            raise ValueError(
                "Invalid number of observations: {}".format(num_observations))
        if weight_decay < 0.0:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay))
        self.theparams = list(params)

        defaults = dict(weight_decay=weight_decay)
        super(Preconditioner, self).__init__(self.theparams, defaults)
        self.num_observations = int(num_observations)
        self.prior_iterations = int(prior_iterations)
        self.rank = int(est_rank)
        self.stepnumber = 0

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        self.lam = torch.zeros(1, device=self.device)
        self.W_var = torch.zeros(1, device=self.device)
        self.alpha = torch.zeros(1, device=self.device)

        self.optim_class = optim_class
        self.optim_hyperparams = optim_hyperparams
        self.lr = lr

        self._InitializeLists()

    # An initialization function, called in the __init__() and at the end of CreateLowRank()
    def _InitializeLists(self):
        for group in self.param_groups:
            self.accumulated_hess_vec = list(
                torch.zeros_like(p) for p in group['params'])

            self.gradient = list(torch.zeros_like(p) for p in group['params'])
            self.accumulated_gradient = list(
                torch.zeros_like(p) for p in group['params'])
            self.vec = list(torch.ones_like(p) for p in group['params'])
            self.STAS = list(torch.zeros(1, device=self.device)
                             for p in group['params'])
            self.STAAS = list(torch.zeros(1, device=self.device)
                              for p in group['params'])
            self.STS = list(torch.zeros(1, device=self.device)
                            for p in group['params'])

            self.S = list(torch.zeros((tuple(p.size()) + (self.num_observations,)),
                                      device=self.device) for p in group['params'])
            self.X = list(torch.zeros((p.numel(), self.num_observations),
                                      device=self.device) for p in group['params'])
            self.Y = list(torch.zeros((tuple(p.size()) + (self.num_observations,)),
                                      device=self.device) for p in group['params'])

            self.STWS = list(torch.zeros(
                (self.num_observations, self.num_observations), device=self.device) for p in group['params'])
            self.STLS = list(torch.zeros((self.num_observations,),
                                         device=self.device) for p in group['params'])
            self.inner_product = list(torch.zeros(
                (self.num_observations, self.num_observations), device=self.device) for p in group['params'])

        self.update_counter = 0

    # ? A helper funtion, it is called only in UpdateEstimatedHessian
    # It does ?

    def HessianVectorProduct(self):
        df_sum = torch.zeros(1, device=self.device)

        for group in self.param_groups:

            for v, g, p in zip(self.vec, self.gradient, group['params']):
                g.data = p.grad.clone()

                df_sum += torch.sum(v * p.grad)

        df_sum.backward()

        for group in self.param_groups:
            weight_decay = group['weight_decay']

            for hv, g, p, v in zip(self.accumulated_hess_vec, self.gradient, group['params'], self.vec):
                hv.data = p.grad.data - g.data + (weight_decay * v.data)

        return

    # A function that uses the data from GatherCurvatureInformation to construct a
    # Prior Estimate for the Hessian
    def EstimatePrior(self):

        sts = 0
        stas = 0
        staas = 0

        for STS, STAS, STAAS in zip(self.STS, self.STAS, self.STAAS):
            sts += STS.item()
            stas += STAS.item()
            staas += STAAS.item()

        self.alpha = torch.tensor([stas / staas], dtype=dtype)

        hv2_temp = torch.zeros(1)  # tensor([0.0])
        ghv_temp = torch.zeros(1)  # tensor([0.0])
        g_temp = torch.zeros(1)  # tensor([0.0])

        n = self.update_counter
        for g, hv in zip(self.accumulated_gradient, self.accumulated_hess_vec):

            hv2_temp.data += (torch.norm(hv)**2)
            ghv_temp.data += torch.sum(g * hv)
            g_temp.data += (torch.norm(g)**2)
            g.div_(n)
            hv.div_(n)

        self.W_var.data = torch.tensor([stas / sts], dtype=dtype)
        self.lam.data = torch.abs(torch.tensor([sts]) - g_temp.data).div_(n)

        print('[Estimate prior] alpha: {:.2e} w: {:.2e} lambda: {:.2e}'.format(
            self.alpha.item(), self.W_var.item(), self.lam.item()))

        self.update_counter = 0
        return

    # This runs first and collects information about the parameter's curvature.

    def GatherCurvatureInformation(self):
        df_sum = torch.zeros(1, device=self.device)  # tensor([0.0])

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            for grad_temp, v, ag, p in zip(self.gradient, self.vec, self.accumulated_gradient, group['params']):
                grad_temp.data = p.grad.clone()
                v.data = grad_temp.data + weight_decay * p.data
                ag.data += v.data
                df_sum += torch.sum(v * p.grad)

        df_sum.backward()

        a_STAS = 0.0
        a_STAAS = 0.0
        i = 0
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            for g, hv, v, p in zip(self.gradient, self.accumulated_hess_vec, self.vec, group['params']):
                hv_temp = p.grad.data - g.data + (weight_decay * v)

                STS = torch.norm(v)**2
                STAS = torch.sum(v * hv_temp)
                STAAS = torch.norm(hv_temp)**2
                # print(STAAS.item(),STAS.item(),STS.item())
                a_STAS += STAS.data
                a_STAAS += STAAS.data
                self.STS[i] += STS
                self.STAS[i] += STAS
                self.STAAS[i] += STAAS
                i += 1
                # print('STAS: {:5.3e}, STAAS: {:5.3e}, STAS/STAAS: {:5.3e}'.format(STAS,STAAS,STAS/STAAS))
                hv.data += hv_temp.data

        self.update_counter += 1

    # Helper function, called only in UpdateEstimatedHessian().
    def ApplyEstimatedInverse(self):
        m = self.update_counter
        for group in self.param_groups:
            w_var = self.W_var.to(self.device)
            alph = self.alpha.to(self.device)
            for S, vec, X, p, ip in zip(self.S, self.vec, self.X, group['params'], self.inner_product):

                # B's dimensionality needs to be adjusted for torch.solve
                B = torch.mv(S[..., :m].view(-1, m).t(),
                             p.grad.view(-1)).view(m, -1)
                A = torch.eye(m, device=self.device) + ip[:m, :m]

                proj = alph ** 2 * w_var ** 2 * torch.mv(X[..., :m].view(-1, m),
                                                         torch.solve(B,
                                                                     A)
                                                         [0][:, 0]).view_as(p.grad)

                vec.data = alph * p.grad - proj
    #                 print(torch.norm(vec))
    #                 print((alph**2*w_var**2).item())

    # The magic happens here. Runs a couple of times after a prior is constructed. It updates the estimate for the Hessian.
    def UpdateEstimatedHessian(self):
        m = self.update_counter


        print("m: ", m)
        w_var = self.W_var.to(self.device)
        lam = self.lam.to(self.device)


        alph = self.alpha.to(self.device)
        # print('---------------')
        if self.update_counter == 0:
            for S, Y, g, hv, STWS, STLS, vec, ip, X in zip(self.S, self.Y, self.accumulated_gradient,
                                                           self.accumulated_hess_vec, self.STWS, self.STLS,
                                                           self.vec, self.inner_product, self.X):

                S_norm = torch.norm(g.mul(-alph)) ** 2
                S.data[..., 0] = g.mul(-alph) / torch.sqrt(S_norm)
                Y.data[..., 0] = hv.mul(-alph) / torch.sqrt(S_norm)
                delta = (Y[..., 0] - S[..., 0]).view(-1)
#                 print(delta)
                # S_norm=torch.sum(S[...,0]**2)
                # print(S_norm)
                STWS.data[0, 0] = w_var * S_norm
                STLS.data[0] = lam * S_norm
#                 print(STWS,STLS)
                X.data[:, 0] = delta / (w_var * STWS[0, 0] + lam * STLS[0])
#                 print('X1: ',X)
#                 print(torch.sum(S[...,0]*X*w_var**2 +1))

#                 vec=alph*w_var
#                 print(X.shape)
                ip[0, 0] = alph * w_var**2 * \
                    torch.sum(X[:, 0] * S[..., 0].view(-1))

        else:
            self.ApplyEstimatedInverse()
            self.HessianVectorProduct()
            for S, Y, hv, STWS, STLS, vec, ip, X in zip(self.S, self.Y,
                                                        self.accumulated_hess_vec, self.STWS,
                                                        self.STLS, self.vec, self.inner_product, self.X):
                #                 print(S.shape,torch.norm(vec))
                S_norm = torch.norm(vec)**2
                Y_norm = torch.norm(hv)**2
                # S.data[...,m]=vec.data
                # Y.data[...,m]=hv.data
                S.data[..., m] = (vec / torch.sqrt(S_norm)).data
                Y.data[..., m] = (hv / torch.sqrt(S_norm)).data

                delta = (Y[..., :m + 1] - alph *
                         S[..., :m + 1]).view(-1, m + 1)
#                 print(delta.shape,delta[0,:])

                # print('m: {:2d} |S|^2: {:1.2e} |Y|^2: {:1.2e} sqrt(|Y|^2/|S|^2): {:1.2e} '.format(self.update_counter,S_norm.item(),Y_norm.item(),(torch.sqrt(Y_norm/S_norm)).item()))
                STWS.data[:m, m] = w_var * \
                    torch.mv(S[..., :m].view(-1, m).t(), S[..., m].view(-1))
                STWS.data[m, :m] = STWS[:m, m]
                STWS.data[m, m] = w_var * S_norm
                # STWS.data = 0.5*(STWS.data+STWS.t().data)
                # print(STWS.data[:m,:m])
                # w_var*S_norm
                # STWS.data[0,0]=w_var*S_norm
                STLS.data[m] = lam * S_norm
                stls = torch.sqrt(STLS[:m + 1])
                # print(STLS[:m+1])
                # print(S_norm)
                # print(stls,STWS[:m+1,:m+1])
                # print((STWS[:m+1,:m+1]/stls)/stls.unsqueeze(1))
                D, V = torch.symeig(
                    (STWS[:m + 1, :m + 1] / stls) / stls.unsqueeze(1), eigenvectors=True)
                # print(D.data)
#                 print('V: ',V,D)
#                 torch.symeig()
#                 print(D,STWS[:m+1,:m+1])
#                 print(STLS)

                F_V = V / stls.unsqueeze(1)
#                 print('F: ',F_V)
#                 print(torch.mm(delta[:2,:],F_V)/torch.sqrt(lam))
                X.data[:, :m + 1] = torch.mm((torch.mm(delta, F_V) / torch.sqrt(lam)) / (
                    w_var / lam * D + 1.0), F_V.t()) / torch.sqrt(lam)
#                 print('X: ',X[0:2,:])
#                 X=delta/(w_var*STWS[0,0]+lam*STLS[0])
#                 print(torch.sum(S[...,0]*X*w_var**2 +1))

#                 print(X.shape)
                ip[:m + 1, :m + 1] = alph * w_var**2 * \
                    torch.mm(X[:, :m + 1].t(), S[..., :m + 1].view(-1, m + 1))

        self.update_counter += 1

    # Runs after an estimate for the Hessian is constructed. Does the preconditioning. Probably a lot of math?
    def CreateLowRank(self):

        for group in self.param_groups:

            w_var = self.W_var.cpu()
            alph = self.alpha.cpu()

            for p, S, X in zip(group['params'], self.S, self.X):
                V = S.view(-1, self.update_counter)
                S_ = S.cpu()
                X_ = X.cpu()

                def Matv(v):
                    return w_var.numpy() ** 2 * X_.numpy().dot(S_[..., :self.update_counter].view(-1, self.update_counter).numpy().T.dot(v)) + v / alph.numpy()

                def rMatv(v):
                    return w_var.numpy() ** 2 * S_[..., :self.update_counter].view(-1, self.update_counter).numpy().dot(X_.numpy().T.dot(v)) + v / alph.numpy()

                effective_rank = np.min([self.rank, X.shape[0] - 1])
                LinOp = splinalg.LinearOperator(
                    (X.shape[0], X.shape[0]), matvec=Matv, rmatvec=rMatv)
                sing_vec, sing_val, _ = splinalg.svds(
                    LinOp, k=effective_rank)  # ,tol=1e-2)
                print(' 1/a: {:.2e} w: {:.2e} sigma: {}'.format(1.0 /
                                                                alph.item(), w_var.item(), np.sqrt(sing_val)))

                nnz = np.count_nonzero(sing_val)
                if nnz < effective_rank:
                    effective_rank = nnz

                shape = list(p.shape)
                shape.append(effective_rank)
                state = self.state[p]
                state['preconditioned_vectors'] = torch.from_numpy(sing_vec[:, :effective_rank].astype(
                    np.float32)).view(shape).to(self.device)  # .view_as()
                state['preconditioned_scaling'] = torch.from_numpy(
                    np.sqrt(sing_val[:effective_rank]).astype(np.float32)).to(self.device)

        self.update_counter = 0
        self._InitializeLists()

    # Called every iteration, first only to estimate the Hessian, then to do the optimizing
    # TODO: keep track of estimation phase internally.

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        if self.stepnumber < self.prior_iterations:
            self.GatherCurvatureInformation()
        elif self.stepnumber == self.prior_iterations:
            self.EstimatePrior()
        elif self.stepnumber < self.prior_iterations + self.num_observations:
            self.UpdateEstimatedHessian()
        elif self.stepnumber == self.prior_iterations + self.num_observations:  # finish the estimate
            self.UpdateEstimatedHessian()
            self.CreateLowRank()

            # initialize the_optimizer
            print(self.lr)
            if self.lr is None:
                self.lr = self.alpha.item()
            print(self.lr)
            self.optim_hyperparams.update(lr=self.lr)
            self.the_optimizer = self.optim_class(
                self.theparams, **self.optim_hyperparams)

        else:  # optim step

            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    grad = p.grad.data

                    if grad.is_sparse:
                        raise RuntimeError(
                            'Method does not support sparse gradients')
                    state = self.state[p]

                    # get the preconditioning results and transform the gradient.
                    U = state['preconditioned_vectors']
                    D = state['preconditioned_scaling']
                    D_min = torch.min(D)
                    effective_rank = len(D)
                    buf = torch.sum(U * (1.0 / D - 1.0) * torch.mv(
                        U.view(-1, effective_rank).t(), grad.view(-1)), dim=-1)  # conditioned gradient
                    p.grad.data = buf.add(grad)

            self.the_optimizer.step()

        self.stepnumber += 1

        return loss
