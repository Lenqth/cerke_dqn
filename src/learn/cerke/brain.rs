use anyhow::Result;

use tch::{
    nn,
    nn::OptimizerConfig,
    nn::{ModuleT, Optimizer, VarStore},
    Device, Tensor,
};

use crate::learn::state_to_feature::{ACTION_SIZE, STATE_SIZE};

fn network(vs: &nn::Path) -> impl ModuleT {
    nn::seq_t()
        .add(nn::linear(
            vs / "layer1",
            STATE_SIZE as i64,
            128,
            Default::default(),
        ))
        .add(nn::batch_norm1d(vs, 128, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(vs, 128, 128, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(vs, 128, ACTION_SIZE as i64, Default::default()))
}

pub trait Brain {
    fn train(&mut self, batch: Vec<(&[f32], &[f32], &[f32])>) -> Result<()>;
    fn forward(&self, batch: Vec<&[f32]>) -> Result<Vec<Vec<f32>>>;
    fn update_hard(&mut self);
    fn update_soft(&mut self, tau: f64);
}

pub struct QNet {
    device: Device,
    vs_learn: VarStore,
    vs_target: VarStore,
    net_learn: Box<dyn ModuleT>,
    net_target: Box<dyn ModuleT>,
    opt: Optimizer,
}

impl QNet {
    pub fn new() -> Self {
        let device = Device::cuda_if_available();

        let vs_learn = nn::VarStore::new(device);
        let net_learn = Box::new(network(&vs_learn.root()));

        let vs_target = nn::VarStore::new(device);
        let net_target = Box::new(network(&vs_target.root()));

        let opt = nn::Adam::default().build(&vs_learn, 0.00025).unwrap();
        Self {
            device,
            vs_learn,
            net_learn,
            vs_target,
            net_target,
            opt,
        }
    }
}

impl Brain for QNet {
    #[must_use]
    fn train(&mut self, batch: Vec<(&[f32], &[f32], &[f32])>) -> Result<()> {
        let dev = self.device;
        let net = &self.net_learn;
        let (batch_size, channels, output_channels) =
            (batch.len(), batch[0].0.len(), batch[0].1.len());

        let mut input = Vec::with_capacity(batch_size * channels);
        let mut output = Vec::with_capacity(batch_size * output_channels);
        let mut mask = Vec::with_capacity(batch_size * output_channels);

        for i in 0..batch_size {
            let _ii = i * channels;

            let batch_item = batch[i].0;
            for j in 0..channels {
                input.push(batch_item[j]);
            }

            let batch_output_item = batch[i].1;
            for j in 0..output_channels {
                output.push(batch_output_item[j]);
            }

            let _batch_mask_item = batch[i].2;
            for j in 0..output_channels {
                mask.push(batch_output_item[j]);
            }
        }
        let input_tensor = Tensor::of_slice(&input)
            .reshape(&[batch_size as i64, channels as i64])
            .to(dev);
        let output_tensor = Tensor::of_slice(&output)
            .reshape(&[batch_size as i64, output_channels as i64])
            .to(dev);
        let mask_tensor = Tensor::of_slice(&mask)
            .reshape(&[batch_size as i64, output_channels as i64])
            .to(dev);

        let res = net.forward_t(&input_tensor, true);
        let res = &mask_tensor * res;
        let loss = Tensor::huber_loss(&res, &output_tensor, tch::Reduction::Sum, 1.0f64);

        //            let regularization = self.vs.trainable_variables().iter().map(|x| x.abs().sum(Kind::Float)).reduce(|x,y| x + y ).unwrap();
        //            let loss = loss + regularization * Scalar::float(1e-7f64);

        self.opt.backward_step(&loss);
        println!("{}", f64::from(&loss));
        Ok(())
    }

    #[must_use]
    fn forward(&self, batch: Vec<&[f32]>) -> Result<Vec<Vec<f32>>> {
        let dev = self.device;
        let net = &self.net_target;
        let (batch_size, channels) = (batch.len(), batch[0].len());

        let mut input = Vec::with_capacity(batch_size * channels);
        for i in 0..batch_size {
            let _ii = i * channels;
            let batch_item = batch[i];
            for j in 0..channels {
                input.push(batch_item[j]);
            }
        }
        let input_tensor = Tensor::of_slice(&input).reshape(&[batch_size as i64, channels as i64]);

        let result = net.forward_t(&input_tensor.to(dev), false);
        Ok(result.into())
    }

    fn update_hard(&mut self) {
        self.vs_target.copy(&self.vs_learn).unwrap();
    }

    fn update_soft(&mut self, _tau: f64) {
        todo!()
    }
}
