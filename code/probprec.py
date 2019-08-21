import numpy as np
import torch
from scipy.sparse import linalg as splinalg

dtype = torch.float

class Preconditioner(torch.optim.Optimizer):

    def __init__(self, params, est_rank=2, num_observations=5, prior_iterations=10, weight_decay=0, lr=None,
                 optim_class=torch.optim.SGD, **optim_hyperparams):

        if not 0 <= weight_decay:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay))

        # init the statedict and default values.
        defaults = dict(est_rank=est_rank, num_observations=num_observations,
                        prior_iterations=prior_iterations,
                        weight_decay=weight_decay,
                        optim_class=optim_class)
        defaults.update(optim_hyperparams)

        super(Preconditioner, self).__init__(list(params), defaults)

        #this is a hack, but needed
        self.device = self.param_groups[0]['params'][0].device

        self.lam = torch.zeros(1, device=self.device)
        self.W_var = torch.zeros(1, device=self.device)
        self.alpha = torch.zeros(1, device=self.device)

        self.optim_class = optim_class
        optim_hyperparams.update(weight_decay=weight_decay)
        self.optim_hyperparams = optim_hyperparams

        self.start_estimate(lr, est_rank, num_observations, prior_iterations)

    # An initialization function, called in start_estimate()
    def _initialize_lists(self):
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


    # initialize the_optimizer
    def _init_the_optimizer(self):
        if self.lr is None:
            self.lr = self.alpha.item()
        self.optim_hyperparams.update(lr=self.lr)
        print("[_init_the_optimizer] Initializing ", self.optim_class.__name__, " with: ", self.optim_hyperparams)
        self.the_optimizer = self.optim_class(
            self.param_groups, **self.optim_hyperparams)

    # start_estimate() resets internal state and starts a new estimation process
    # Called in the init, but can/should also be used externally.
    # by default, discards the saved lr and keeps the other hyperparams.

    def start_estimate(self, lr=None,
                       est_rank=None,
                       num_observations=None,
                       prior_iterations=None):

        self.lr = lr

        if num_observations is not None:
            if not 0 <= num_observations:
                raise ValueError(
                    "Invalid number of observations: {}".format(num_observations))
            self.num_observations = int(num_observations)

        if prior_iterations is not None:
            if not 0 <= prior_iterations:
                raise ValueError(
                    "Invalid number of prior iterations: {}".format(prior_iterations))
            self.prior_iterations = int(prior_iterations)

        if est_rank is not None:
            if not 0 <= est_rank:
                raise ValueError("Invalid Hessian rank: {}".format(est_rank))
            self.est_rank = int(est_rank)

        self.stepnumber = 0
        self.prior_counter = 0
        self.update_counter = 0

        self._initialize_lists()

    # logs a norm of the gradient and alpha
    def get_log(self):
        #log alpha
        #log gradnorm
        gradnorm = 0
        for group in self.param_groups:
            for p in group['params']:
                gradnorm += torch.sum(torch.pow(p.grad.data, 2))
        gradnorm = torch.sqrt(gradnorm)
        #return for deepOBS to write
        return gradnorm.item()
################################################################################
##                                                                            ##
##                MATH FUNCTIONS                                              ##
##                                                                            ##
################################################################################

    # This runs first and collects information about the parameter's curvature.
    def _gather_curvature_information(self):
        df_sum = torch.zeros(1, device=self.device)  # tensor([0.0])

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            for g, v, ag, p in zip(self.gradient, self.vec, self.accumulated_gradient, group['params']):
                g.data = p.grad.clone()
                v.data = g.data + weight_decay * p.data
                ag.data += v.data
                df_sum += torch.sum(v * p.grad)

        df_sum.backward()

        i = 0
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            for g, hv, v, p in zip(self.gradient, self.accumulated_hess_vec, self.vec, group['params']):
                hv_temp = p.grad.data - g.data + (weight_decay * v)

                STS = torch.norm(v)**2
                STAS = torch.sum(v * hv_temp)
                STAAS = torch.norm(hv_temp)**2 # see eq. 11, "A" means Lambda

                print('[_gather_curvature_information] STAS =', STAS)
                self.STS[i] += STS
                self.STAS[i] += STAS
                self.STAAS[i] += STAAS
                i += 1
                hv.data += hv_temp.data

        self.prior_counter += 1

    # A function that uses the data from _gather_curvature_information to construct
    # a Prior Estimate for the Hessian
    def _estimate_prior(self):

        sts = sum(self.STS)
        stas = sum(self.STAS)
        staas = sum(self.STAAS)

        g_temp = torch.zeros(1)

        n = self.prior_counter

        for g, hv in zip(self.accumulated_gradient, self.accumulated_hess_vec):
            g_temp.data += (torch.norm(g)**2)
            g.div_(n)
            hv.div_(n)

        self.alpha = torch.tensor([stas / staas], dtype=dtype)
        self.W_var.data = torch.tensor([stas / sts], dtype=dtype)
        self.lam.data = torch.abs(torch.tensor([sts]) - g_temp.data).div_(n)

        print("[_estimate_prior] (sums) sts:", sts, "stas", stas, "staas", staas)
        print('[_estimate_prior] alpha: {:.2e} w: {:.2e} lambda: {:.2e}'.format(
            self.alpha.item(), self.W_var.item(), self.lam.item()))

    def _setup_estimated_hessian(self):
        w_var = self.W_var.to(self.device)
        lam = self.lam.to(self.device)
        alph = self.alpha.to(self.device)

        for S, Y, g, hv, STWS, STLS, vec, ip, X in zip(self.S, self.Y, self.accumulated_gradient,
                                                       self.accumulated_hess_vec, self.STWS, self.STLS,
                                                       self.vec, self.inner_product, self.X):
            S_norm = torch.norm(g.mul(-alph)) ** 2
            S.data[..., 0] = g.mul(-alph) / torch.sqrt(S_norm)
            Y.data[..., 0] = hv.mul(-alph) / torch.sqrt(S_norm)
            delta = (Y[..., 0] - S[..., 0]).view(-1)
            STWS.data[0, 0] = w_var * S_norm
            STLS.data[0] = lam * S_norm
            X.data[:, 0] = delta / (w_var * STWS[0, 0] + lam * STLS[0])
            ip[0, 0] = alph * w_var**2 * \
                torch.sum(X[:, 0] * S[..., 0].view(-1))

        self.update_counter += 1  # set to 1

    # Some math function
    def _apply_estimated_inverse(self):
        m = self.update_counter
        w_var = self.W_var.to(self.device)
        alph = self.alpha.to(self.device)

        for group in self.param_groups:
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

    # Computes the Hessian-vector products of the true Hessian of the params and
    # the vectors self.vec
    # Result is saved in hv
    def _hessian_vector_product(self):
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

    # The magic happens here. Runs a couple of times after a prior is constructed.
    # It updates the estimate for the Hessian.
    # TODO: refactor
    def _update_estimated_hessian(self):
        m = self.update_counter

        #print("[_update_estimated_hessian] m: ", m)
        w_var = self.W_var.to(self.device)
        lam = self.lam.to(self.device)
        alph = self.alpha.to(self.device)

        for S, Y, hv, STWS, STLS, vec, ip, X in zip(self.S, self.Y,
                                                    self.accumulated_hess_vec, self.STWS,
                                                    self.STLS, self.vec, self.inner_product, self.X):
            S_norm = torch.norm(vec)**2
            S.data[..., m] = (vec / torch.sqrt(S_norm)).data
            Y.data[..., m] = (hv / torch.sqrt(S_norm)).data

            delta = (Y[..., :m + 1] - alph *
                     S[..., :m + 1]).view(-1, m + 1)

            STWS.data[:m, m] = w_var * \
                torch.mv(S[..., :m].view(-1, m).t(), S[..., m].view(-1))
            STWS.data[m, :m] = STWS[:m, m]
            STWS.data[m, m] = w_var * S_norm
            STLS.data[m] = lam * S_norm

            stls = torch.sqrt(STLS[:m + 1])

            D, V = torch.symeig( #D: eigenvalues, V: eigenvectors
                (STWS[:m + 1, :m + 1] / stls) / stls.unsqueeze(1), eigenvectors=True)
            F_V = V / stls.unsqueeze(1)

            X.data[:, :m + 1] = torch.mm((torch.mm(delta, F_V) / torch.sqrt(lam)) / (
                w_var / lam * D + 1.0), F_V.t()) / torch.sqrt(lam)

            ip[:m + 1, :m + 1] = alph * w_var**2 * \
                torch.mm(X[:, :m + 1].t(), S[..., :m + 1].view(-1, m + 1))

        self.update_counter += 1

    # Runs after an estimate for the Hessian is constructed.
    # Creates the parts needed to construct the Preconditioner
    # The numpy calculations are done on the CPU
    # TODO: maybe get rid of numpy? Nah.

    def _create_low_rank(self):
        w_var = self.W_var.cpu()
        alph = self.alpha.cpu()

        for group in self.param_groups:
            for p, S, X in zip(group['params'], self.S, self.X):
                V = S.view(-1, self.update_counter)
                S_ = S.cpu()
                X_ = X.cpu()

                def Matv(v):
                    return w_var.numpy() ** 2 * X_.numpy().dot(S_[..., :self.update_counter].view(-1, self.update_counter).numpy().T.dot(v)) + v / alph.numpy()

                def rMatv(v):
                    return w_var.numpy() ** 2 * S_[..., :self.update_counter].view(-1, self.update_counter).numpy().dot(X_.numpy().T.dot(v)) + v / alph.numpy()

                effective_rank = np.min([self.est_rank, X_.shape[0] - 1])
                LinOp = splinalg.LinearOperator((X_.shape[0], X_.shape[0]),
                                                matvec=Matv,
                                                rmatvec=rMatv)
                sing_vec, sing_val, _ = splinalg.svds(LinOp,
                                                      k=effective_rank)  # ,tol=1e-2)
                print('[_create_low_rank] sigma: {}'.format(np.sqrt(sing_val)) )

                effective_rank = min(effective_rank, np.count_nonzero(sing_val))

                shape = list(p.shape)
                shape.append(effective_rank)
                state = self.state[p]
                state['preconditioned_vectors'] = torch.from_numpy(sing_vec[:, :effective_rank].astype(
                    np.float32)).view(shape).to(self.device)  # .view_as()
                state['preconditioned_scaling'] = torch.from_numpy(
                    np.sqrt(sing_val[:effective_rank]).astype(np.float32)).to(self.device)

    # Applies the Preconditioner to the gradient
    def _apply_preconditioner(self):
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

################################################################################
##                                                                            ##
##                STEP FUNCTION                                               ##
##                                                                            ##
################################################################################

    # Called every iteration, first only to estimate the Hessian
    # then to do the optimizing
    # Main logic happens here
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        # Make prior estimate
        if self.stepnumber < self.prior_iterations:
            self._gather_curvature_information()  # uses new data
            if self.stepnumber == self.prior_iterations - 1:
                self._estimate_prior()            # does not use new data
                self._setup_estimated_hessian()  # does not use new data
        # Make posterier estimate (pseudocode in the paper)
        elif self.stepnumber < self.prior_iterations + self.num_observations - 1:
            self._apply_estimated_inverse()        # uses new data
            self._hessian_vector_product()         # uses new data
            self._update_estimated_hessian()       # does not directly use new data
        elif self.stepnumber == self.prior_iterations + self.num_observations - 1:
            self._create_low_rank()                # uses new data
            self._init_the_optimizer()           # does not use new data
        else:
            self._apply_preconditioner()
            self.the_optimizer.step()


        self.stepnumber += 1

        return loss
