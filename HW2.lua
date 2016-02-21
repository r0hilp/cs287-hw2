-- Only requirements allowed
require("hdf5")
require("nn")
require("optim")

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', 'PTB.hdf5', 'data file')
cmd:option('-action', 'train', 'train or test')
cmd:option('-classifier', 'nb', 'classifier to use')
cmd:option('-window_size', 5, 'window size')
cmd:option('-warm_start', '', 'torch file with previous model')
cmd:option('-test_model', '', 'model to test on')
cmd:option('-model_out_name', 'train', 'output file name of model')
cmd:option('-optim_method', 'sgd', 'loss function optimization method')

-- Hyperparameters
cmd:option('-use_cap', 1, 'use capitalization')
cmd:option('-alpha', 1, 'alpha for naive Bayes')
cmd:option('-eta', 0.01, 'learning rate for SGD')
cmd:option('-batch_size', 32, 'batch size for SGD')
cmd:option('-max_epochs', 20, 'max # of epochs for SGD')
cmd:option('-lambda', 1.0, 'regularization lambda for SGD')
cmd:option('-hidden', 300, 'size of hidden layer for neural network')
cmd:option('-cap_vec', 5, 'size of capitalization features vector embedding')
cmd:option('-use_glove', 0, 'use word embeddings from Glove')

function window_features(X, nfeat)
  local range = torch.range(0, (window_size - 1)*nfeat, nfeat):long():view(1, window_size)
  X:add(range:expand(X:size(1), X:size(2)))
  return X
end

function train_nb(X, X_cap, Y)
  -- Trains naive Bayes model
  local alpha = opt.alpha
  local N = X:size(1)

  local timer = torch.Timer()
  local time = timer:time().real

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

  print('Time for naive Bayes:', (timer:time().real - time) * 1000, 'ms')
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

function compute_err(Y, pred)
  -- Compute error from Y
  local _, argmax = torch.max(pred, 2)
  argmax:squeeze()

  local correct
  if Y then
    correct = argmax:eq(Y:long()):sum()
  end
  return argmax, correct
end

function eval(X, X_cap, Y, W, W_cap, b)
  -- Returns error from Y
  local pred = linear(X, X_cap, W, W_cap, b)
  
  local argmax, correct = compute_err(Y, pred)
  return argmax, correct
end

function linear_model()
  if opt.warm_start ~= '' then
    return torch.load(opt.warm_start).model
  end

  -- linear logistic model
  local model = nn.Sequential()
  local embeds = nn.ParallelTable()
  local word_embed = nn.LookupTable(nfeatures * window_size, nclasses)
  word_embed.weight[1]:zero()
  local cap_embed = nn.LookupTable(ncapfeatures * window_size, nclasses)
  cap_embed.weight[1]:zero()
  embeds:add(word_embed)
  embeds:add(cap_embed)
  model:add(embeds)
  model:add(nn.JoinTable(2)) -- 2 for batch

  model:add(nn.Sum(2))
  model:add(nn.LogSoftMax())

  return model
end

function neural_model(word_vecs)
  if opt.warm_start ~= '' then
    return torch.load(opt.warm_start).model
  end

  -- neural network from Collobert
  local model = nn.Sequential()
  local embeds = nn.ParallelTable()
  local word_embed = nn.LookupTable(nfeatures * window_size, vec_size)
  word_embed.weight[1]:zero()
  local cap_embed = nn.LookupTable(ncapfeatures * window_size, opt.cap_vec)
  cap_embed.weight[1]:zero()
  if opt.use_glove == 1 then
    word_embed.weight = word_vecs:repeatTensor(window_size, 1)
  end
  embeds:add(word_embed)
  embeds:add(cap_embed)
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

function model_eval(model, criterion, X, X_cap, Y)
    -- batch eval
    model:evaluate()
    local N = X:size(1)
    local batch_size = opt.batch_size

    local total_loss = 0
    local total_correct = 0
    for batch = 1, X:size(1), batch_size do
        local sz = batch_size
        if batch + batch_size > N then
          sz = N - batch + 1
        end
        local X_batch = X:narrow(1, batch, sz)
        local X_cap_batch = X_cap:narrow(1, batch, sz)
        local Y_batch = Y:narrow(1, batch, sz)

        local outputs = model:forward{X_batch, X_cap_batch}
        local loss = criterion:forward(outputs, Y_batch)

        local _, correct = compute_err(Y_batch, outputs)
        total_correct = total_correct + correct
        total_loss = total_loss + loss * batch_size
    end

    return total_loss / N, total_correct / N
end

function train_model(X, X_cap, Y, valid_X, valid_X_cap, valid_Y, word_vecs)
  local eta = opt.eta
  local batch_size = opt.batch_size
  local max_epochs = opt.max_epochs
  --local lambda = opt.lambda
  local N = X:size(1)

  local model
  if opt.classifier == 'logreg' then
    model = linear_model()
  else
    model = neural_model(word_vecs)
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
  local state = { learningRate = eta } -- weightDecay = lambda

  local prev_loss = 1e10
  local epoch = 1
  local timer = torch.Timer()
  while epoch <= max_epochs do
      print('Epoch:', epoch)
      local epoch_time = timer:time().real
      local total_loss = 0

      -- loop through each batch
      model:training()
      for batch = 1, N, batch_size do
          --if ((batch - 1) / batch_size) % 1000 == 0 then
            --print('Sample:', batch)
            --print('Current train loss:', total_loss / batch)
            --print('Current time:', 1000 * (timer:time().real - epoch_time), 'ms')
          --end
          local sz = batch_size
          if batch + batch_size > N then
            sz = N - batch + 1
          end
          local X_batch = X:narrow(1, batch, sz)
          local X_cap_batch = X_cap:narrow(1, batch, sz)
          local Y_batch = Y:narrow(1, batch, sz)

          -- closure to return err, df/dx
          local func = function(x)
            -- get new parameters
            if x ~= params then
              params:copy(x)
            end
            -- reset gradients
            grads:zero()

            -- forward
            local outputs = model:forward{ X_batch, X_cap_batch }
            local loss = criterion:forward(outputs, Y_batch)

            -- track errors
            total_loss = total_loss + loss * batch_size
            --total_correct = total_correct + correct

            -- compute gradients
            local df_do = criterion:backward(outputs, Y_batch)
            model:backward({ X_batch, X_cap_batch }, df_do)

          --model:get(1):get(1).gradWeight[1]:zero()
          --model:get(1):get(2).gradWeight[1]:zero()

            return loss, grads
          end

          if opt.optim_method == 'sgd' then
            optim.sgd(func, params, state)
          elseif opt.optim_method == 'adadelta' then
            optim.adadelta(func, params, state)
          elseif opt.optim_method == 'adagrad' then
            optim.adagrad(func, params, state)
          end
          -- padding to zero
          model:get(1):get(1).weight[1]:zero()
          model:get(1):get(2).weight[1]:zero()
      end

      print('Train loss:', total_loss / N)

      local loss, valid_percent = model_eval(model, criterion, valid_X, valid_X_cap, valid_Y)
      print('Valid loss:', loss)
      print('Valid percent:', valid_percent)

      print('time for one epoch: ', (timer:time().real - epoch_time) * 1000, 'ms')
      print('')
      if loss > prev_loss and epoch > 5 then
        prev_loss = loss
        break
      end
      prev_loss = loss
      epoch = epoch + 1
      torch.save(opt.model_out_name .. '_' .. opt.classifier .. '.t7', { model = model })
  end
  print('Trained', epoch, 'epochs')
  return model, prev_loss
end

function main() 
   -- Parse input params
   opt = cmd:parse(arg)
   local f = hdf5.open(opt.datafile, 'r')
   nclasses = f:read('nclasses'):all():long()[1]
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
   local test_ids = f:read('test_ids'):all():long()
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
   if opt.action == 'train' then
    print('Training...')
    local W, W_cap, b
    local model
    if opt.classifier == 'nb' then
        W, W_cap, b = train_nb(X_win, X_cap_win, Y)
    else
        model = train_model(X_win, X_cap_win, Y, valid_X_win, valid_X_cap_win, valid_Y, word_vecs)
    end

    -- Validate.
    local pred, percent
    local loss
    if opt.classifier == 'nb' then
      pred, percent = eval(valid_X_win, valid_X_cap_win, valid_Y, W, W_cap, b)
    else
      loss, percent = model_eval(model, nn.ClassNLLCriterion(), valid_X_win, valid_X_cap_win, valid_Y)
    end
    print('Percent correct:', percent)

    local log_f = io.open(opt.classifier .. '.log', 'w')
    log_f:write('Error ', percent, '\n')
    for k, v in pairs(opt) do
      log_f:write(k, ' ', v, '\t')
    end

   -- Test.
   elseif opt.action == 'test' then
    print('Testing...')
    local W, W_cap, b
    local model = torch.load(opt.test_model).model
    local outputs = model:forward{test_X_win, test_X_cap_win}
    local _, pred = torch.max(outputs, 2)
    --local test_pred = eval(test_X_win, test_X_cap_win, valid_Y, W, W_cap, b)
    f = io.open('PTB_pred.test', 'w')
    f:write("ID,Category\n")
    for i = 1, test_X:size(1) do
      f:write(test_ids[i], ",", pred[i][1],"\n")
    end
   end
end

main()
