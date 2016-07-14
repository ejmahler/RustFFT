/// Given an array of size width * height, representing a flattened 2D array,
/// transpose the rows and columns of that 2D array into the output
pub fn transpose<T>(width: usize, height: usize, input: &[T], output: &mut [T])
    where T: Copy
{
    assert_eq!(width * height, input.len());
    assert_eq!(input.len(), output.len());

    for y in 0..height {
        for x in 0..width {
            let input_index = x + y * width;
            let output_index = y + x * height;

            unsafe {
                *output.get_unchecked_mut(output_index) = *input.get_unchecked(input_index);
            }
        }
    }
}


#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_transpose() {
        let input_list = vec![
            (2, 2, vec![
                1, 2,
                3, 4,
            ]),
            (3, 2, vec![
                1, 2, 3,
                4, 5, 6
            ]),
            (2, 3, vec![
                1, 2,
                3, 4,
                5, 6,

            ]),
        ];
        let expected_list = vec![
            vec![
                1, 3,
                2, 4,
            ],
            vec![
                1, 4,
                2, 5,
                3, 6
            ],
            vec![
                1, 3, 5,
                2, 4, 6,
            ],
        ];

        for (input, expected) in input_list.iter().zip(expected_list.iter()) {

            let (width, height, ref input_array) = *input;

            let mut output = input_array.clone();
            transpose(width, height, input_array.as_slice(), output.as_mut_slice());

            assert_eq!(expected.as_slice(), output.as_slice());
        }
    }
}