-- Only requirements allowed
require("hdf5")
require("nn")
require("optim")

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', 'PTB.hdf5', 'data file')
cmd:option('-classifier', 'nb', 'classifier to use')
cmd:option('-window_size', 5, 'window size')

-- Hyperparameters
cmd:option('-alpha', 1, 'alpha for naive Bayes')
cmd:option('-eta', 0.01, 'learning rate for SGD')
cmd:option('-batch_size', 32, 'batch size for SGD')
cmd:option('-max_epochs', 20, 'max # of epochs for SGD')
cmd:option('-lambda', 1.0, 'regularization lambda for SGD')
cmd:option('-hidden', 100, 'size of hidden layer for neural network')

function window_features(X, nfeat)
  local range = torch.range(0, (window_size - 1)*nfeat, nfeat):long():view(1, window_size)
  X:add(range:expand(X:size(1), X:size(2)))
  return X
end

function train_nb(X, X_cap, Y, alpha)
  -- Trains naive Bayes model
  alpha = alpha or 0
  local N = X:size(1)

  -- intercept
  local b = torch.histc(Y:double(), nclasses)
  local b_logsum = torch.log(b:sum())
  b:log():csub(b_logsum)

  local W = torch.Tensor(nclasses, window_size * nfeatures):fill(alpha)
  for i = 1, N do
    W:select(1, Y[i]):indexAdd(1, X[i], torch.ones(X[i]:size(1)))
  end
  -- zero out padding counts
  W:select(2, 1):zero()
  local W_logsum = torch.log(W:sum(2))
  W:log():csub(W_logsum:expand(W:size(1), W:size(2)))
  -- padding weight to zero
  W:select(2, 1):zero()

  -- cap features
  local W_cap = torch.Tensor(nclasses, window_size * ncapfeatures):fill(alpha)
  for i = 1, N do
    W_cap:select(1, Y[i]):indexAdd(1, X_cap[i], torch.ones(X_cap[i]:size(1)))
  end
  -- zero out padding counts
  W_cap:select(2, 1):zero()
  local W_cap_logsum = torch.log(W_cap:sum(2))
  W_cap:log():csub(W_cap_logsum:expand(W_cap:size(1), W_cap:size(2)))
  -- padding weight to zero
  W_cap:select(2, 1):zero()

  return W, W_cap, b
end

function linear(X, X_cap, W, W_cap, b)
  local N = X:size(1)
  local z = torch.zeros(N, nclasses)
  for i = 1, N do
    z[i]:add(b)
    z[i]:add(W:index(2, X[i]):sum(2))
    z[i]:add(W_cap:index(2, X_cap[i]):sum(2))
  end

  return z
end

function eval(X, X_cap, Y, W, W_cap, b)
  -- Returns error from Y
  --
  local pred = linear(X, X_cap, W, W_cap, b)
  
  -- Compute error from Y
  local _, argmax = torch.max(pred, 2)
  argmax:squeeze()

  local err
  if Y then
    err = argmax:eq(Y:long()):sum()
    err = err / Y:size(1)
  end
  return argmax, err
end

function linear_model()
  -- linear logistic model
  local model = nn.Sequential()
  model:add(nn.LookupTable(nfeatures, opt.vec_size))
  model:add(nn.Linear(opt.vec_size, nclasses))
  model:add(nn.LogSoftMax())

  return model
end

function neural_model()
  -- neural network from Collobert
  local model = nn.Sequential()
  model:add(nn.LookupTable(nfeatures, opt.vec_size))
  model:add(nn.Linear(opt.vec_size, opt.hidden))
  model:add(nn.HardTanh())
  model:add(nn.Linear(opt.hidden, nclasses))
  model:add(nn.LogSoftMax())

  return model
end

function train_model(X, Y, eta, batch_size, max_epochs, lambda, valid_X, valid_Y, model_type)
  eta = eta or 0
  batch_size = batch_size or 0
  max_epochs = max_epochs or 0

  local model
  if model_type == 'logreg' then
    model = linear_model()
  else
    model = neural_model()
  end

  local criterion = nn.ClassNLLCriterion()

  -- shuffle for batches
  local shuffle = torch.randperm(N):long()
  X = X:index(1, shuffle)
  Y = Y:index(1, shuffle)

  -- only call this once
  local params, grads = model:getParameters()
  -- sgd state
  local state = { learningRate = eta, weightDecay = lambda }

  local prev_loss = 1e10
  while epoch < max_epochs do
      local total_err = 0

      -- loop through each batch
      for batch = 1, X:size(1), batch_size do
          if ((batch - 1) / batch_size) % 100 == 0 then
            print('Sample:', batch)
          end
          local sz = batch_size
          if batch + batch_size > N then
            sz = N - batch + 1
          end
          local X_batch = X:narrow(1, batch, sz)
          local Y_batch = Y:narrow(1, batch, sz)

          model:training()

          -- closure to return err, df/dx
          local func = function(x)
            -- get new parameters
            if x ~= params then
              params:copy(x)
            end
            -- reset gradients
            grads:zero()

            -- forward
            local err = criterion:forward(model:forward(X_batch), Y_batch)

            -- track errors
            total_err = total_err + err * batch_size

            -- compute gradients
            local df_do = criterion:backward(outputs, Y_batch)
            model:backward(X_batch, df_do)

            return err, grads
          end

          optim.sgd(func, params, state)
          -- padding to zero
          --model:get(1).weight[1]:zero()
      end

      -- calculate loss
      model:evaluate()

      local loss = 0
      for batch = 1, valid_X:size(1), batch_size do
          if ((batch - 1) / batch_size) % 100 == 0 then
            print('Sample:', batch)
          end
          local sz = batch_size
          if batch + batch_size > N then
            sz = N - batch + 1
          end
          local X_batch = valid_X:narrow(1, batch, sz)
          local Y_batch = valid_Y:narrow(1, batch, sz)

          local l = criterion:forward(model:forward(X_batch), Y_batch)
          loss = loss + l * batch_size
      end

      print('Epoch:', epoch)
      print('Train err:', err / X:size(1))
      print('Valid err:', loss / valid_X:size(1))

      if torch.abs(prev_loss - loss) / prev_loss < 0.001 then
        prev_loss = loss
        break
      end
      prev_loss = loss
      epoch = epoch + 1
      torch.save('train.t7', { model = model })
  end
  print('Trained', epoch, 'epochs')
  return model, prev_loss
end

function main() 
   -- Parse input params
   opt = cmd:parse(arg)
   local f = hdf5.open(opt.datafile, 'r')
   nclasses = f:read('nclasses'):all():long()[1] - 1
   nfeatures = f:read('nfeatures'):all():long()[1]
   --ncapfeatures = f:read('ncapfeatures'):all():long()[1]
   ncapfeatures = 5
   window_size = opt.window_size

   print('Loading data...')
   local X = f:read('train_input'):all():long()
   local X_cap = f:read('train_cap_input'):all():long()
   local Y = f:read('train_output'):all()
   local valid_X = f:read('valid_input'):all():long()
   local valid_X_cap = f:read('valid_cap_input'):all():long()
   local valid_Y = f:read('valid_output'):all()
   local test_X = f:read('test_input'):all():long()
   local test_X_cap = f:read('test_cap_input'):all():long()
   local word_vecs = f:read('word_vecs'):all()

   local X_win = window_features(X, nfeatures)
   local X_cap_win = window_features(X_cap, ncapfeatures)
   local valid_X_win = window_features(valid_X, nfeatures)
   local valid_X_cap_win = window_features(valid_X_cap, ncapfeatures)
   local test_X_win = window_features(test_X, nfeatures)
   local test_X_cap_win = window_features(test_X_cap, ncapfeatures)

   -- Train.
   print('Training...')
   local W, W_cap, b = train_nb(X_win, X_cap_win, Y, opt.alpha)

   -- Test.
   local pred, err = eval(valid_X_win, valid_X_cap_win, valid_Y, W, W_cap, b)
   print('Percent correct:', err)
end

main()
