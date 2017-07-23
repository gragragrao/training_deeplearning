class SGD:
    def __init__(self, learn_rate=0.01):
        self.learn_rate = learn_rate

    def update(self, params, grads):
        for param, grad in zip(params, grads):
            param.assign_add(-self.learn_rate * grad)


class Momentum:
    def __init__(self, learn_rate=0.01, momentum=0.9):
        self.learn_rate = learn_rate
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = [np.zeros_like(param) for param in params]

        for i in range(len(self.v)):
            self.v[i] = self.momentum * self.v[i] - self.learn_rate * grads[i]
            params[i] += self.v[i]


class AdaGrad:
    def __init__(self, learn_rate=0.01, params):
        self.learn_rate = learn_rate
        self.h = [tf.Variable(np.zeros_like(param).astype('float32')) for param in params]
        self.params = params

    def update(self, grads):
        for h, param, grad in zip(self.h, self.params, grads):
            h.assign_add(grad * grad)
            param.assign_add(-self.learn_rate * np.sqrt(1 / (h + 1e-7)) * grad)


class Adam:
    def __init__(self, alpha=10**(-3), beta=0.9, gamma=0.999, params):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.v = [np.zeros_like(param).astype('float32') for param in params]
        self.r = [np.zeros_like(param).astype('float32') for param in params]
        self.params = params

    # t は試行回数
    def update(self, grads, t):
        for v, r, param, grad in zip(self.v, self.r, self.params, grads):
            v.assign(self.beta * v + (1 - self.beta) * grad)
            r.assign(self.gamma * r + (1 - self.gamma) * grad * grad)
            param.assign_add((-self.alpha * v) / ((np.sqrt(r / (1 - self.gamma ** t))) * (1 - self.beta ** t)))


# 更新の方法
def sgd(cost, params, eps=np.float32(0.1)):
    g_params = tf.gradients(cost, params)

    updates = []
    for param, g_param in zip(params, g_params):
        if g_param is not None:
            updates.append(param.assign_add(-eps * g_param))
    return updates


v = tf.Variable(np.zeros(in_dim, out_dim).astype('float32'), name='v')
r = tf.Variable(np.zeros(in_dim, out_dim).astype('float32'), name='r')


def adam(cost, params, v, r, t, alpha=10**(-3), beta=0.9, gamma=0.999):
    grads = tf.gradients(cost, params)

    updates = []
    for param, grad in zip(params, grads):
        if grad is not None:
            updates.append(v.assign(beta * v + (1 - beta) * grad))
            updates.append(r.assign(gamma * r + (1 - gamma) * grad * grad))
            updates.append(param.assign_add((-alpha * v) / ((np.sqrt(r / (1 - gamma ** t)) + 1e-8) * (1 - beta ** t))))

    return updates













###
