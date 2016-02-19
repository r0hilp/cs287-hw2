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
cmd:option('-use_cap', 1, 'use capitalization')
cmd:option('-alpha', 1, 'alpha for naive Bayes')
cmd:option('-eta', 0.01, 'learning rate for SGD')
cmd:option('-batch_size', 32, 'batch size for SGD')
cmd:option('-max_epochs', 20, 'max # of epochs for SGD')
cmd:option('-lambda', 1.0, 'regularization lambda for SGD')
cmd:option('-hidden', 100, 'size of hidden layer for neural network')
cmd:option('-cap_vec', 5, 'size of capitalization features vector embedding')

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
  local W_cap
  if opt.use_cap == 1 then
    W_cap = torch.Tensor(nclasses, window_size * ncapfeatures):fill(alpha)
    for i = 1, N do
      W_cap:select(1, Y[i]):indexAdd(1, X_cap[i], torch.ones(X_cap[i]:size(1)))
    end
    -- zero out padding counts
    W_cap:select(2, 1):zero()
    local W_cap_logsum = torch.log(W_cap:sum(2))
    W_cap:log():csub(W_cap_logsum:expand(W_cap:size(1), W_cap:size(2)))
    -- padding weight to zero
    W_cap:select(2, 1):zero()
  end

  return W, W_cap, b
end

function linear(X, X_cap, W, W_cap, b)
  local N = X:size(1)
  local z = torch.zeros(N, nclasses)
  for i = 1, N do
    z[i]:add(b)
    z[i]:add(W:index(2, X[i]):sum(2))
    if opt.use_cap == 1 then
      z[i]:add(W_cap:index(2, X_cap[i]):sum(2))
    end
  end

  return z
end

function eval(X, X_cap, Y, W, W_cap, b)
  -- Returns error from Y
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
  local embeds = nn.ParallelTable()
  embeds:add(nn.LookupTable(nfeatures * window_size, nclasses))
  embeds:add(nn.LookupTable(ncapfeatures * window_size, nclasses))
  model:add(embeds)
  
  model:add(nn.JoinTable(2)) -- 2 for batch
  model:add(nn.Sum(2))
  model:add(nn.LogSoftMax())

  return model
end

function neural_model()
  -- neural network from Collobert
  local model = nn.Sequential()
  local embeds = nn.ParallelTable()
  embeds:add(nn.LookupTable(nfeatures * window_size, vec_size))
  embeds:add(nn.LookupTable(ncapfeatures * window_size, opt.cap_vec))
  model:add(embeds)
  
  local view = nn.ParallelTable()
  view:add(nn.View(vec_size * window_size)) -- concat
  view:add(nn.View(opt.cap_vec * window_size))
  model:add(view)

  model:add(nn.JoinTable(2)) -- 2 for batch
  model:add(nn.Linear((vec_size + opt.cap_vec) * window_size, opt.hidden))
  model:add(nn.HardTanh())
  model:add(nn.Linear(opt.hidden, nclasses))
  model:add(nn.LogSoftMax())

  return model
end

function model_eval(model, criterion, X, X_cap, Y, batch_size)
    -- batch eval
    model:evaluate()
    local N = X:size(1)

    local loss = 0
    for batch = 1, X:size(1), batch_size do
        if ((batch - 1) / batch_size) % 100 == 0 then
          print('Sample:', batch)
        end
        local sz = batch_size
        if batch + batch_size > N then
          sz = N - batch + 1
        end
        local X_batch = X:narrow(1, batch, sz)
        local X_cap_batch = X_cap:narrow(1, batch, sz)
        local Y_batch = Y:narrow(1, batch, sz)

        local l = criterion:forward(model:forward{X_batch, X_cap_batch}, Y_batch)
        loss = loss + l * batch_size
    end

    return loss
end

function train_model(X, X_cap, Y, valid_X, valid_X_cap, valid_Y, eta, batch_size, max_epochs, lambda, model_type)
  eta = eta or 0
  batch_size = batch_size or 0
  max_epochs = max_epochs or 0
  model_type = model_type or 'logreg'
  local N = X:size(1)

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
  X_cap = X_cap:index(1, shuffle)
  Y = Y:index(1, shuffle)

  -- only call this once
  local params, grads = model:getParameters()
  -- sgd state
  local state = { learningRate = eta, weightDecay = lambda }

  local prev_loss = 1e10
  local epoch = 0
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
          local X_cap_batch = X_cap:narrow(1, batch, sz)
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
            local outputs = model:forward{X_batch, X_cap_batch}
            local err = criterion:forward(outputs, Y_batch)

            -- track errors
            total_err = total_err + err * batch_size

            -- compute gradients
            local df_do = criterion:backward(outputs, Y_batch)
            model:backward({X_batch, X_cap_batch}, df_do)

            return err, grads
          end

          optim.sgd(func, params, state)
          -- padding to zero
          model:get(1):get(1).weight[1]:zero()
          model:get(1):get(2).weight[1]:zero()
      end

      print('Epoch:', epoch)
      print('Train err:', total_err / X:size(1))

      local loss = model_eval(model, criterion, valid_X, valid_X_cap, valid_Y, batch_size)
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
   -- Word embeddings from glove
   local word_vecs = f:read('word_vecs'):all()
   vec_size = word_vecs:size(2)

   local X_win = window_features(X, nfeatures)
   local X_cap_win = window_features(X_cap, ncapfeatures)
   local valid_X_win = window_features(valid_X, nfeatures)
   local valid_X_cap_win = window_features(valid_X_cap, ncapfeatures)
   local test_X_win = window_features(test_X, nfeatures)
   local test_X_cap_win = window_features(test_X_cap, ncapfeatures)

   -- Train.
   print('Training...')
   local W, W_cap, b
   local model
   if opt.classifier == 'nb' then
      W, W_cap, b = train_nb(X_win, X_cap_win, Y, opt.alpha)
   else
      model = train_model(X_win, X_cap_win, Y, valid_X_win, valid_X_cap_win, valid_Y, opt.eta, opt.batch_size, opt.max_epochs, opt.lambda, opt.classifier)
    end

   -- Test.
   local pred, err
   if opt.classifier == 'nb' then
     pred, err = eval(valid_X_win, valid_X_cap_win, valid_Y, W, W_cap, b)
   else
     err = model_eval(model, nn.ClassNLLCriterion(), valid_X_win, valid_X_cap_win, valid_Y, opt.batch_size)
   end
   print('Percent correct:', err)

   --local test_pred = eval(test_X_win, test_X_cap_win, valid_Y, W, W_cap, b)
   --f = io.open('PTB_pred.test', 'w')
   --f:write("ID,Category\n")
   --for i = 1, test_X:size(1) do
     --f:write(i, ",", pred[i][1],"\n")
   --end

end

main()
