fn super_resolution() -> anyhow::Result<()> {
    let env = infer::Environment::new()?;
    let sess = env.new_session("super-resolution-10.onnx")?;

    let input = ndarray_image::open_image("obama.png", ndarray_image::Colors::Luma)?;
    let input = input.mapv(|e| e as f32) * (1.0_f32 / 255.0_f32);
    let input = infer::InputTensor {
        data: input.as_slice().unwrap(),
        shape: &[1, 1, 224, 224],
    };

    let output = sess.run(vec![("input", input)])?;

    let asdf = output[0]
        .1
        .as_slice()
        .expect("reee")
        .iter()
        .map(|x| (x * 255_f32) as u8)
        .collect::<Vec<u8>>();
    let arr = ndarray::ArrayView::from_shape((672, 672, 1), &asdf)?;

    ndarray_image::save_image("test_out.png", arr, ndarray_image::Colors::Luma)?;
    Ok(())
}

fn main() {
    super_resolution().unwrap();
    println!("Success!");
}
