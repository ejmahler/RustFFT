
use num_traits::{Zero, One, FromPrimitive, PrimInt, Signed};
use std::mem::swap;

pub fn primitive_root(prime: u64) -> Option<u64> {
    let test_exponents: Vec<u64> = distinct_prime_factors(prime - 1)
        .iter()
        .map(|factor| (prime - 1) / factor)
        .collect();
    'next: for potential_root in 2..prime {
        // for each distinct factor, if potential_root^(p-1)/factor mod p is 1, reject it
        for exp in &test_exponents {
            if modular_exponent(potential_root, *exp, prime) == 1 {
                continue 'next;
            }
        }

        // if we reach this point, it means this root was not rejected, so return it
        return Some(potential_root);
    }
    None
}

/// computes base^exponent % modulo using the standard exponentiation by squaring algorithm
pub fn modular_exponent<T: PrimInt>(mut base: T, mut exponent: T, modulo: T) -> T {
    let one = T::one();

    let mut result = one;

    while exponent > Zero::zero() {
        if exponent & one == one {
            result = result * base % modulo;
        }
        exponent = exponent >> One::one();
        base = (base * base) % modulo;
    }

    result
}

pub fn multiplicative_inverse<T: PrimInt + FromPrimitive>(a: T, n: T) -> T {
    // we're going to use a modified version extended euclidean algorithm
    // we only need half the output

    let mut t = Zero::zero();
    let mut t_new = One::one();

    let mut r = n;
    let mut r_new = a;

    while r_new > Zero::zero() {
        let quotient = r / r_new;

        r = r - quotient * r_new;
        swap(&mut r, &mut r_new);

        // t might go negative here, so we have to do a checked subtract
        // if it underflows, wrap it around to the other end of the modulo
        // IE, 3 - 4 mod 5  =  -1 mod 5  =  4
        let t_subtract = quotient * t_new;
        t = if t_subtract < t {
            t - t_subtract
        } else {
            n - (t_subtract - t) % n
        };
        swap(&mut t, &mut t_new);
    }

    t
}

pub fn extended_euclidean_algorithm<T: PrimInt + Signed + FromPrimitive>(a: T,
                                                                                   b: T)
                                                                                   -> (T, T, T) {
    let mut s = Zero::zero();
    let mut s_old = One::one();

    let mut t = One::one();
    let mut t_old = Zero::zero();

    let mut r = b;
    let mut r_old = a;

    while r > Zero::zero() {
        let quotient = r_old / r;

        r_old = r_old - quotient * r;
        swap(&mut r_old, &mut r);

        s_old = s_old - quotient * s;
        swap(&mut s_old, &mut s);

        t_old = t_old - quotient * t;
        swap(&mut t_old, &mut t);
    }

    (r_old, s_old, t_old)
}

/// return all of the prime factors of n, but omit duplicate prime factors
pub fn distinct_prime_factors(mut n: u64) -> Vec<u64> {
    let mut result = Vec::new();

    // handle 2 separately so we dont have to worry about adding 2 vs 1
    if n % 2 == 0 {
        while n % 2 == 0 {
            n /= 2;
        }
        result.push(2);
    }
    if n > 1 {
        let mut divisor = 3;
        let mut limit = (n as f32).sqrt() as u64 + 1;
        while divisor < limit {
            if n % divisor == 0 {

                // remove as many factors as possible from n
                while n % divisor == 0 {
                    n /= divisor;
                }
                result.push(divisor);

                // recalculate the limit to reduce the amount of work we need to do
                limit = (n as f32).sqrt() as u64 + 1;
            }

            divisor += 2;
        }

        if n > 1 {
            result.push(n);
        }
    }

    result
}

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub struct PrimeFactor {
    pub value: usize,
    pub count: usize,
}

/// Factors an integer into its prime factors.
pub fn prime_factors(mut n: usize) -> Vec<PrimeFactor> {
    let mut result = Vec::new();

    let trailing_zeros = n.trailing_zeros();
    if trailing_zeros > 0 {
        result.push(PrimeFactor { value: 2, count: trailing_zeros as usize });
        n >>= trailing_zeros;
    }
    if n > 1 {
        let mut divisor = 3;
        let mut limit = (n as f32).sqrt() as usize + 1;
        while divisor < limit {
            let mut count = 0;
            while n % divisor == 0 {
                n /= divisor;
                count += 1;
            }

            if count > 0 {
                result.push(PrimeFactor { value: divisor, count });
            }

            // recalculate the limit to reduce the amount of other factors we need to check
            limit = (n as f32).sqrt() as usize + 1;
            divisor += 2;
        }

        if n > 1 {
            result.push(PrimeFactor { value: n, count: 1 });
        }
    }

    result
}

// Splits a set of prime factors into two different sets so that the products of the two sets are as close as possible
pub fn partition_factors(mut factors: Vec<PrimeFactor>) -> (Vec<PrimeFactor>, Vec<PrimeFactor>) {
    // Make sure this isn't a prime number
    assert!(factors.len() > 1 || factors[0].count > 0);

    // If the given length is a perfect square, put the square root into both returned arays
    if factors.iter().all(|factor| factor.count % 2 == 0) {
        for factor in factors.iter_mut() {
            factor.count /= 2;
        }
        (factors.clone(), factors)
    } else if factors.len() == 1 {
        // If there's only one factor, just split it as evenly as possible
        let half = vec![PrimeFactor { value: factors[0].value, count: factors[0].count / 2 }];
        factors[0].count -= half[0].count;
        (factors, half)
    } else {
        let mut other_factors = Vec::with_capacity(factors.len());
        let mut this_product = 1;
        let mut other_product = 1;

        factors.retain(|factor| {
            let factor_product = factor.value.pow(factor.count as u32);
            if this_product <= other_product {
                this_product *= factor_product;
                true
            } else {
                other_factors.push(*factor);
                other_product *= factor_product;
                false
            }
        });

        (factors, other_factors)
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn test_modular_exponent() {
        // make sure to test something that would overflow under ordinary circumstances
        // ie 3 ^ 416788 mod 47
        let test_list = vec![
			((2,8,300), 256),
			((2,9,300), 212),
			((1,9,300), 1),
			((3,416788,47), 8),
		];

        for (input, expected) in test_list {
            let (base, exponent, modulo) = input;

            let result = modular_exponent(base, exponent, modulo);

            assert_eq!(result, expected);
        }
    }

    #[test]
    fn test_multiplicative_inverse() {
        let prime_list = vec![3, 5, 7, 11, 13, 17, 19, 23, 29];

        for modulo in prime_list {
            for i in 2..modulo {
                let inverse = multiplicative_inverse(i, modulo);

                assert_eq!(i * inverse % modulo, 1);
            }
        }
    }

    #[test]
    fn test_extended_euclidean() {
        let test_list = vec![
            ((3,5), (1, 2, -1)),
            ((15,12), (3, 1, -1)),
            ((16,21), (1, 4, -3)),
        ];

        for (input, expected) in test_list {
            let (a, b) = input;

            let result = extended_euclidean_algorithm(a, b);
            assert_eq!(expected, result);

            let (gcd, mut a_inverse, mut b_inverse) = result;

            // sanity check: if gcd=1, then a*a_inverse mod b should equal 1 and vice versa
            if gcd == 1 {
                if a_inverse < 0 {
                    a_inverse += b;
                }
                if b_inverse < 0 {
                    b_inverse += a;
                }

                assert_eq!(1, a * a_inverse % b);
                assert_eq!(1, b * b_inverse % a);
            }
        }
    }

    #[test]
    fn test_primitive_root() {
        let test_list = vec![(3, 2), (7, 3), (11, 2), (13, 2), (47, 5), (7919, 7)];

        for (input, expected) in test_list {
            let root = primitive_root(input).unwrap();

            assert_eq!(root, expected);
        }
    }

    #[test]
    fn test_distinct_prime_factors() {
        let test_list = vec![
			(46, vec![2,23]),
			(2, vec![2]),
			(3, vec![3]),
			(162, vec![2, 3]),
			];

        for (input, expected) in test_list {
            let factors = distinct_prime_factors(input);

            assert_eq!(factors, expected);
        }
    }

    #[test]
    fn test_prime_factors() {
        let test_list = vec![
			(2, vec![PrimeFactor { value: 2, count: 1 }]),
			(3, vec![PrimeFactor { value: 3, count: 1 }]),
			(6, vec![PrimeFactor { value: 2, count: 1 }, PrimeFactor { value: 3, count: 1 }]),
			(256, vec![PrimeFactor { value: 2, count: 8 }]),
			(768, vec![PrimeFactor { value: 2, count: 8 }, PrimeFactor { value: 3, count: 1 }]),
			(44100, vec![PrimeFactor { value: 2, count: 2 }, PrimeFactor { value: 3, count: 2 }, PrimeFactor { value: 5, count: 2 }, PrimeFactor { value: 7, count: 2 }]),
			];

        for (input, expected) in test_list {
            let factors = prime_factors(input);

            assert_eq!(factors, expected);
        }

        for n in 1..200 {
            let factors = prime_factors(n);
            let multiplied : usize = factors.into_iter().map(|factor| factor.value.pow(factor.count as u32)).product();
            assert_eq!(multiplied, n);
        }
    }

    #[test]
    fn test_partition_factors() {

        for n in 4..200 {
            let factors = prime_factors(n);
            if factors.len() > 1 || factors[0].count > 1 {
                let (left_factors, right_factors) = partition_factors(factors);

                let left_product : usize = left_factors.into_iter().map(|factor| factor.value.pow(factor.count as u32)).product();
                let right_product : usize = right_factors.into_iter().map(|factor| factor.value.pow(factor.count as u32)).product();

                assert_eq!(left_product * right_product, n);
            }
        }
    }
}
