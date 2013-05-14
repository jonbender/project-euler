#!/usr/bin/env python
import time
import itertools
import functools 
import collections
import math
import random

def gcd(a,b):
        return a if not b else gcd(b,a % b)

def lcm(a,b):
    return a*b / gcd(a,b)

def miller_rabin(n,k=100):
    '''Probabilistic primality test'''
    prime = True
    composite = False
    def miller_rabin_inner(n,d,s,k):
        for i_k in xrange(k):
            a = random.randint(2,n-2)
            x = pow(a,d,n)
            if x == 1 or x == (n-1):
                continue
            flag = False
            for i_s in xrange(s-1):
                x = pow(x,2,n)
                if x == 1:
                    return composite 
                if x == (n-1):
                    flag = True
                    break
            if flag:
                continue
            else:
                return composite 
        return prime 
    
    if n == 1:
        return composite
    if n <= 3:
        return prime 
    if not n % 2:
        return composite 

    s = 0
    d = n - 1
    while not d % 2:
        s += 1
        d >>= 1
    return miller_rabin_inner(n,d,s,k)

def pollard_rho(n):
    if n == 1:
        return None
    if not n % 2:
        return 2
    c = random.randrange(2,1000000) 
    f = lambda x : (x*x + c) % n
    x,y,d=2,2,1
    while d == 1:
        x = f(x)
        y = f(f(y))
        d = gcd(abs(x-y),n)
    if d == n:
        return None
    else:
        return d

def factors(x):

    if x == 1:
        return []
    fact = []
    tries = 0
    while tries < 3:
        if miller_rabin(x):
            fact.append(x)
            break
        d = pollard_rho(x)
        if d and d > 1:
            tries = 0
            if not miller_rabin(d):
                subfacts = filter(lambda x: x > 1,factors(d))
                fact.extend(subfacts)
                for f in subfacts:
                    #print '%d in f' % x
                    while not x % f and x > f:
                        x /= f
            else:
                fact.append(d)
                while not x % d:
                    x /= d
        else:
            tries +=1
    return fact


def gen_prime():
    '''Eli Bendersky's Erasthenes implemenation'''
    q = 2
    D = {}
    while True:
        if not q in D:
            yield q
            D[q * q] = [q]
        else:
            for p in D[q]:
                D.setdefault(p + q,[]).append(p)
            del D[q]
        q+=1

def gen_tri():
    n = 0
    while True:
        n += 1
        yield n*(n+1)/2 

def gen_fib():
    a = 0
    b = 1 
    while True:
        _tmp = b
        b += a
        a = _tmp
        yield b 

def ex1():
    return sum(filter(lambda x : x % 3 == 0 or x % 5 == 0,xrange(1000)))

def ex2():
    gf = gen_fib()
    even = lambda x : x % 2 == 0
    val = 0
    s = 0
    while val < 4e6:
        if even(val):
            s += val
        val = gf.next()
    return s

def ex3():

    target=600851475143
    n = target
    i = 2
    while i*i < n:
        while n % i == 0:
            n /= i
        i+=1
    return n
    #return sorted(largest_primes,key=max)

def ex4():
    def is_palindrome(x):
        elems = int((math.log10(x)))
        candidate = 0
        _x = x
        for n in xrange(elems+1): 
            prefix = _x / 10 ** (elems - n)
            candidate += prefix * 10 ** n
            _x -= prefix * 10 ** (elems - n)
        return x == candidate

    def is_palindrome_str(x):
        _xs = str(x)
        return _xs == _xs[::-1]

    combos = sorted(itertools.combinations(xrange(100,1000),2),key=lambda x: x[0]*x[1],reverse=True)
    for x,y in combos:
        p = x*y
        if is_palindrome_str(p):
            print (x,y)
            return p

def ex5():
    n = 1
    for i in xrange(1,21):
        n = lcm(n,i)
    return n

def ex6():
    MAX = 100
    sum_sq = sum(map(lambda x : x ** 2,xrange(1,MAX+1)))
    sq_sum = sum(xrange(1,MAX+1)) ** 2 
    return sq_sum - sum_sq

def ex7():
    gp = gen_prime()
    i = 0
    val = 0
    while i < 10001:
        val = gp.next()
        i+=1
    return val

def ex8():
    BIG_STRING='7316717653133062491922511967442657474235534919493496983520312774506326239578318016984801869478851843858615607891129494954595017379583319528532088055111254069874715852386305071569329096329522744304355766896648950445244523161731856403098711121722383113622298934233803081353362766142828064444866452387493035890729629049156044077239071381051585930796086670172427121883998797908792274921901699720888093776657273330010533678812202354218097512545405947522435258490771167055601360483958644670632441572215539753697817977846174064955149290862569321978468622482839722413756570560574902614079729686524145351004748216637048440319989000889524345065854122758866688116427171479924442928230863465674813919123162824586178664583591245665294765456828489128831426076900422421902267105562632111110937054421750694165896040807198403850962455444362981230987879927244284909188845801561660979191338754992005240636899125607176060588611646710940507754100225698315520005593572972571636269561882670428252483600823257530420752963450'
    max_val = -1
    for i in xrange(len(BIG_STRING)-5):
        nums = map(int,BIG_STRING[i:i+5])
        product = reduce(lambda x,y: x*y,nums,1)
        if product > max_val:
            max_val = product
    return max_val 

def ex9():
    for a in xrange(1,1000):
        for b in xrange(a,1000-a):
           c = 1000-a-b
           if c**2 == a**2 + b**2:
               return reduce(lambda x,y: x*y,(a,b,c),1)

def ex10():
    gp = gen_prime()
    return sum(itertools.takewhile(lambda x : x < 2e6,gp))

def ex11():
    GRID = ([[8,2,22,97,38,15,0,40,0,75,4,5,7,78,52,12,50,77,91,8], 
    [49,49,99,40,17,81,18,57,60,87,17,40,98,43,69,48,4,56,62,0], 
    [81,49,31,73,55,79,14,29,93,71,40,67,53,88,30,3,49,13,36,65], 
    [52,70,95,23,4,60,11,42,69,24,68,56,1,32,56,71,37,2,36,91], 
    [22,31,16,71,51,67,63,89,41,92,36,54,22,40,40,28,66,33,13,80], 
    [24,47,32,60,99,3,45,2,44,75,33,53,78,36,84,20,35,17,12,50], 
    [32,98,81,28,64,23,67,10,26,38,40,67,59,54,70,66,18,38,64,70], 
    [67,26,20,68,2,62,12,20,95,63,94,39,63,8,40,91,66,49,94,21], 
    [24,55,58,5,66,73,99,26,97,17,78,78,96,83,14,88,34,89,63,72], 
    [21,36,23,9,75,0,76,44,20,45,35,14,0,61,33,97,34,31,33,95], 
    [78,17,53,28,22,75,31,67,15,94,3,80,4,62,16,14,9,53,56,92], 
    [16,39,5,42,96,35,31,47,55,58,88,24,0,17,54,24,36,29,85,57], 
    [86,56,0,48,35,71,89,7,5,44,44,37,44,60,21,58,51,54,17,58], 
    [19,80,81,68,5,94,47,69,28,73,92,13,86,52,17,77,4,89,55,40], 
    [4,52,8,83,97,35,99,16,7,97,57,32,16,26,26,79,33,27,98,66], 
    [88,36,68,87,57,62,20,72,3,46,33,67,46,55,12,32,63,93,53,69], 
    [4,42,16,73,38,25,39,11,24,94,72,18,8,46,29,32,40,62,76,36], 
    [20,69,36,41,72,30,23,88,34,62,99,69,82,67,59,85,74,4,36,16], 
    [20,73,35,29,78,31,90,1,74,31,49,71,48,86,81,16,23,57,5,54], 
    [1,70,54,71,83,51,54,69,16,92,33,48,61,43,52,1,89,19,67,48]])
    def max_prod(i,j):
        vals = [] 
        if i+3 < 20:
            #Down
            vals.append(get_prod(i,j,1,0))
            if j+3 < 20:
                #Diag down right
                vals.append(get_prod(i,j,1,1))
        if j+3 < 20:
            #Right
            vals.append(get_prod(i,j,0,1))
            if i-3 >= 0:
                #Diag up right
                vals.append(get_prod(i,j,-1,1))
        return 1 if len(vals) == 0 else max(vals)
    def get_prod(i,j,di,dj):
        val = 1
        for l in xrange(4):
            val *= GRID[i+l*di][j+l*dj]
        return val
    m_val = 0
    for i in xrange(20):
        for j in xrange(20):
            m_val = max(m_val,max_prod(i,j))
    return m_val

def ex12():
    #triangle(n) = n*(n+1)/2 

    def get_divisors(x):
        count=0
        #Divisors are mirrored around sqrt n (do we undercount in the case of perfect squares?)
        for d in xrange(2,int(x**0.5)):
            if x % d == 0:
                count+=2
        return count
    gt = gen_tri()
    t = next(gt,None)
    while get_divisors(t) < 500:
        t = next(gt,None)
    return t

def ex13():
    s = 0
    with open('ex13.txt','r') as f:
        nums = map(lambda x : int(x.strip()),f.readlines())
    for num in nums:
        s+=num
    return int(str(s)[:10])

def ex14():
    def gen_collatz(n):
        yield n
        while n != 1:
            if not n % 2:
                n /= 2 
            else:
                n = 3*n + 1
            yield n
    history = {}
    max_len = -1
    max_val = None
    for i in xrange(1,int(1e6)):
        gc = gen_collatz(i)
        seq_len = 1 
        next_val = next(gc,None) 
        while next_val != 1:
            if next_val in history:
                seq_len += history[next_val]
                break
            next_val = next(gc,None) 
            seq_len += 1
        if seq_len > max_len:
            max_val = i
            max_len = seq_len
        history[i] = seq_len
    return max_val 

def ex15():
    def fact(n):
        if n == 0:
            return 1
        return n * fact(n-1)
    def pascal(n,k):
        return fact(n) / (fact(k) * fact(n-k))
    def get_lattice_count(n):
        _val = 0
        for k in xrange(n+1):
            _val += pascal(n,k) ** 2
        return _val
    return get_lattice_count(20)

def ex16():
    '''2^15 = 32768 and the sum of its digits is 3 + 2 + 7 + 6 + 8 = 26.
What is the sum of the digits of the number 2^1000?'''
    dig_sum = sum([int(c) for c in str(2**1000)])
    return dig_sum

def ex17():
    '''If the numbers 1 to 5 are written out in words: one, two, three, four, five, then there are 3 + 3 + 5 + 4 + 4 = 19 letters used in total.
If all the numbers from 1 to 1000 (one thousand) inclusive were written out in words, how many letters would be used?'''
    num_lookup = {
            0:'',1:'one',2:'two',3:'three',4:'four',5:'five',6:'six',7:'seven',8:'eight',9:'nine',10:'ten',11:'eleven',12:'twelve',13:'thirteen',14:'fourteen',
            15:'fifteen',16:'sixteen',17:'seventeen',18:'eighteen',19:'nineteen',20:'twenty',30:'thirty',40:'forty',50:'fifty',60:'sixty',70:'seventy',80:'eighty',
            90:'ninety',1000:'onethousand'
            }
    def get_number(n):
        if n in num_lookup:
            return num_lookup[n]
        elif n < 100:
            ones = n % 10
            tens = n - ones
            s_ten = num_lookup[tens]
            s_one = get_number(ones) 
            return s_ten + s_one
        else:
            tens = n % 100
            hundreds = n - tens
            s_hundred = num_lookup[hundreds / 100] + 'hundred'
            s_ten = get_number(tens) 
            if not s_ten or len(s_ten) == 0:
                return s_hundred
            else:
                return s_hundred + 'and' + s_ten

    letters = 0
    for n in xrange(1,1001):
        letters += len(get_number(n))
    return letters

def ex18():
    P = [[75],
    [95,64],
    [17,47,82],
    [18,35,87,10],
    [20,04,82,47,65],
    [19,01,23,75,03,34],
    [88,02,77,73,07,63,67],
    [99,65,04,28,06,16,70,92],
    [41,41,26,56,83,40,80,70,33],
    [41,48,72,33,47,32,37,16,94,29],
    [53,71,44,65,25,43,91,52,97,51,14],
    [70,11,33,28,77,73,17,78,39,68,17,57],
    [91,71,52,38,17,14,91,43,58,50,27,29,48],
    [63,66,04,68,89,53,67,30,73,16,69,87,40,31],
    [04,62,98,27,23,9,70,98,73,93,38,53,60,04,23]]

    def get_optimal_paths(vec):
        return [max(vec[n],vec[n+1]) for n in xrange(len(vec)-1)]
 
    prev_best = get_optimal_paths(P[-1])
    for n in xrange(len(P)-2,-1,-1):
        new_best = [x[0] + x[1] for x in zip(P[n],prev_best)]
        prev_best = get_optimal_paths(new_best)
    return new_best[0]

def ex19():
    '''You are given the following information, but you may prefer to do some research for yourself.
1 Jan 1900 was a Monday.
Thirty days has September,
April, June and November.
All the rest have thirty-one,
Saving February alone,
Which has twenty-eight, rain or shine.
And on leap years, twenty-nine.
A leap year occurs on any year evenly divisible by 4, but not on a century unless it is divisible by 400.
How many Sundays fell on the first of the month during the twentieth century (1 Jan 1901 to 31 Dec 2000)?'''
    def days_in_month(m,y):
        month_lookup = {1:31,2:28,3:31,4:30,5:31,6:30,7:31,8:31,9:30,10:31,11:30,12:31}
        return 29 if (leap_year(y) and m == 2) else month_lookup[m]
    def leap_year(y):
        return (y % 4 == 0) and ((not ((y % 100) == 0)) or ((y % 400) == 0))
    def days_in_year(y):
        return 366 if leap_year(y) else 365
    def month_day_in_year(d,y):
        m = 0
        while d > 0:
            m += 1
            d_m = days_in_month(m,y)
            if d <= d_m:
                break
            d -= d_m
        return m,d
    sundays = []
    year = 1901
    last_sunday = 364
    first_month_sundays = 0
    while year < 2001:
        first_sunday = last_sunday - days_in_year(year-1)+7
        sundays = xrange(first_sunday,days_in_year(year)+1,7)
        first_month_vec = filter(lambda x: month_day_in_year(x,year)[1] == 1,sundays)
        first_month_sundays += len(first_month_vec)
        last_sunday = sundays[-1]
        year += 1
    return first_month_sundays

def ex20():
    '''n! means n  (n  1)  ...  3  2  1
For example, 10! = 10  9  ...  3  2  1 = 3628800,
and the sum of the digits in the number 10! is 3 + 6 + 2 + 8 + 8 + 0 + 0 = 27.
Find the sum of the digits in the number 100!'''

    def fact(n):
        if n == 0:
            return 1
        return n * fact(n-1)
    return sum([int(n) for n in str(fact(100))])

def ex21():
    '''Let d(n) be defined as the sum of proper divisors of n (numbers less than n which divide evenly into n).
If d(a) = b and d(b) = a, where a b, then a and b are an amicable pair and each of a and b are called amicable numbers.
For example, the proper divisors of 220 are 1, 2, 4, 5, 10, 11, 20, 22, 44, 55 and 110; therefore d(220) = 284. The proper divisors of 284 are 1, 2, 4, 71 and 142; so d(284) = 220.  
Evaluate the sum of all the amicable numbers under 10000.'''

    def d_old(n):
        _val = 0
        for i in xrange(1,int(n/2) + 1):
                if n % i == 0:
                    print 'Got divisor: %d' % i
                    _val += i
        return _val
    def d(n):
        _val = 0
        for i in xrange(1,int(n ** 0.5)):
            if n % i == 0:
                j = n / i
                _val += i
                if j != i and j < n:
                    _val += j
        return _val
    t0 = time.time()
    sum_divs = map(d,xrange(1,10001))
    amicable = set()
    for i,b in enumerate(sum_divs):
        a = i+1
        if a != b and b-1 < len(sum_divs) and sum_divs[b-1] == a:
            amicable.add(a)
            amicable.add(b)
     
    print 'Time taken: %f' % (time.time() - t0)
    return sum(amicable)

def ex22():

    '''Using names.txt (right click and 'Save Link/Target As...'), a 46K text file containing over five-thousand first names, begin by sorting it into alphabetical order. Then working out the alphabetical value for each name, multiply this value by its alphabetical position in the list to obtain a name score.
For example, when the list is sorted into alphabetical order, COLIN, which is worth 3 + 15 + 12 + 9 + 14 = 53, is the 938th name in the list. So, COLIN would obtain a score of 938  53 = 49714.
What is the total of all the name scores in the file?'''
    import string
    alookup = dict(zip(string.ascii_uppercase,xrange(1,len(string.ascii_uppercase)+1)))
    with open('names.txt','r') as f:
        sorted_names = sorted(map(lambda x: x.strip().replace('"',''),f.read().split(',')))
    def name_score(name):
        return reduce(lambda x,y: x+alookup[y],name,0)

    return sum([(i+1)*name_score(name) for i,name in enumerate(sorted_names)])

def ex23():
    '''A perfect number is a number for which the sum of its proper divisors is exactly equal to the number. For example, the sum of the proper divisors of 28 would be 1 + 2 + 4 + 7 + 14 = 28, which means that 28 is a perfect number.
A number n is called deficient if the sum of its proper divisors is less than n and it is called abundant if this sum exceeds n.
As 12 is the smallest abundant number, 1 + 2 + 3 + 4 + 6 = 16, the smallest number that can be written as the sum of two abundant numbers is 24. By mathematical analysis, it can be shown that all integers greater than 28123 can be written as the sum of two abundant numbers. However, this upper limit cannot be reduced any further by analysis even though it is known that the greatest number that cannot be expressed as the sum of two abundant numbers is less than this limit.
Find the sum of all the positive integers which cannot be written as the sum of two abundant numbers.'''
    limit = 28123
    limit_offset = limit - 12
    def divs(n):
        _divs = set()
        for i in xrange(1,int(n ** 0.5)+1):
            if n % i == 0:
                j = n / i
                _divs.add(i)
                if j != i and j < n:
                    _divs.add(j)
        return _divs
    def abundant_gen():
        abundants = filter(lambda x : sum(divs(x)) > x,xrange(1,limit))
        for i,n in enumerate(abundants):
            j = i
            s = n + abundants[j] 
            while s <= limit and j < len(abundants):
                yield s
                s = n + abundants[j] 
                j+=1
    abundant_sums = set(abundant_gen())
    #abundant_sums = set(filter(lambda x : x <= limit,map(sum,itertools.combinations(abundants,2))))
    #for n in abundants:
        #abundant_sums.add(2*n)
    _acc = 0
    for i in xrange(1,limit+1):
        if not i in abundant_sums:
            _acc += i
    return _acc

def ex24():
    def perms(x):
        if len(x) == 0:
            yield []
        if len(x) == 1:
            yield x
        for i,x_i in enumerate(x):
            x_intermed = perms(x[:i] + x[i+1:])
            x_l = next(x_intermed,None)
            while x_l:
                yield [x_i] + x_l 
                x_l = next(x_intermed,None)
    p = perms(list(xrange(10)))
    for n in xrange(1000000):
        perm = next(p,None)
    return ''.join(map(str,perm))

def ex25():
    def first_num(n):
        gf = gen_fib()
        i = 1
        f = 1
        while f and f < 10 ** (n-1):
            i+=1
            f = next(gf,None)
        return i

def ex26():
    '''A unit fraction contains 1 in the numerator. The decimal representation of the unit fractions with denominators 2 to 10 are given:
1/2	= 	0.5
1/3	= 	0.(3)
1/4	= 	0.25
1/5	= 	0.2
1/6	= 	0.1(6)
1/7	= 	0.(142857)
1/8	= 	0.125
1/9	= 	0.(1)
1/10	= 	0.1
Where 0.1(6) means 0.166666..., and has a 1-digit recurring cycle. It can be seen that 1/7 has a 6-digit recurring cycle.
Find the value of d 1000 for which 1/d contains the longest recurring cycle in its decimal fraction part.'''

    def long_div_gen(a,b):
        while True:
            zeros = 0 
            while b > a:
                a *= 10
                zeros+=1
            for n in xrange(1,zeros):
                yield 0
            yield a / b
            if a % b == 0:
                break
            a -= b * (a / b)

    def firstn(a,b,n,):
        ldg = long_div_gen(a,b)
        return filter(lambda x : x != None, [next(ldg,None) for i in xrange(n)])

    def ldd_floyd(a,b):
        RANGE=5000
        max_limit = 100
        lim = 0
        mu = 0
        digits = firstn(a,b,RANGE)
        if len(digits) < RANGE:
            return None,None
        i_t = 0
        i_h = 1
        h = digits[i_h]
        t = digits[i_t]
        result_found = False
        while not result_found and i_h < RANGE:
            h = digits[i_h]
            t = digits[i_t]
            if h == t:
                lam = i_h - i_t
                if digits[i_t:i_h+1] == digits[i_h:i_h+lam+1]:
                    result_found = True
                    return i_t,lam
            i_t += 1
            i_h += 2
        return None,None

    max_div = None
    max_val = -1
    for n in xrange(1,1001):
        mu,lam = ldd_floyd(1,n)
        if lam > max_val:
            max_div,max_val = lam,n
        #if mu is not None:
            #print 'For 1/%d, got (mu,lambda) -> (%d,%d)' % (n,mu,lam)
        #else:
            #print 'No results for 1/%d' % n 
    return max_div,max_val

def ex27():
    '''Euler published the remarkable quadratic formula:

n**2 + n + 41

It turns out that the formula will produce 40 primes for the consecutive values n = 0 to 39. However, when n = 40, 402 + 40 + 41 = 40(40 + 1) + 41 is divisible by 41, and certainly when n = 41, 41**2 + 41 + 41 is clearly divisible by 41.

Using computers, the incredible formula  n**2  79n + 1601 was discovered, which produces 80 primes for the consecutive values n = 0 to 79. The product of the coefficients, 79 and 1601, is 126479.

Considering quadratics of the form:

n**2 + an + b, where |a|  1000 and |b|  1000

where |n| is the modulus/absolute value of n
e.g. |11| = 11 and |4| = 4
Find the product of the coefficients, a and b, for the quadratic expression that produces the maximum number of primes for consecutive values of n, starting with n = 0.'''

    def quad(n,a,b):
        return n**2 + a*n + b
    gp = gen_prime()
    primes = set()
    last_prime = gp.next()
    primes.add(last_prime)
    most_primes = (-1,-1,-1)
    for a in xrange(-1000,1001):
        for b in xrange(-1000,1000):
            n = 0
            prime_found = True
            prime_count = 0 
            while prime_found:
                q = quad(n,a,b)
                n +=1
                while last_prime < q:
                    last_prime = gp.next()
                    primes.add(last_prime)
                if q in primes:
                    prime_count +=1
                    prime_found = True
                else:
                    prime_found = False
            if prime_count > most_primes[0]:
                most_primes= (prime_count,a,b)
    return most_primes

def ex28():
    diag_sum = 1 
    for n in xrange(3,1002,2):
        top_right = n**2
        diags = map(lambda x: top_right - x*(n-1),xrange(4))
        diag_sum += sum(diags)
    return diag_sum

def ex29():
    terms = set()
    for a in xrange(2,101):
        for b in xrange(2,101):
            terms.add(a**b)
    return len(terms)

def ex30():
    '''Surprisingly there are only three numbers that can be written as the sum of fourth powers of their digits:
    1634 = 14 + 64 + 34 + 44
    8208 = 84 + 24 + 04 + 84
    9474 = 94 + 44 + 74 + 44
    As 1 = 14 is not a sum it is not included.
The sum of these numbers is 1634 + 8208 + 9474 = 19316.
Find the sum of all the numbers that can be written as the sum of fifth powers of their digits.'''
    s = 0
    #How to identify the max value it can take without trial & error?
    for n in xrange(2,500000):
        if n == sum([int(c) ** 5 for c in str(n)]):
            s+=n
    return s

def ex31():
    '''In England the currency is made up of pound, L, and pence, p, and there are eight coins in general circulation:
1p, 2p, 5p, 10p, 20p, 50p, L1 (100p) and L2 (200p).
It is possible to make L2 in the following way:
1L1 + 150p + 220p + 15p + 12p + 31p
How many different ways can L2 be made using any number of coins?'''

    purse = (1,2,5,10,20,50,100,200)
    purse = purse[::-1] 
    A = map(lambda x : [0] * 201, xrange(len(purse) + 1))
    A[0][0] = 1
    for i in xrange(8):
        for j in xrange(201):
            for k in xrange(j,201,purse[i]):
                A[i+1][k]+=A[i][j]
    return A[8][200]

    #I don't understand this... :/
    #for a in xrange(initial_amount,-1,-200):
        #for b in xrange(a,0,-100):
            #for c in xrange(b,0,-50):
                #for d in xrange(c,0,-20):
                    #for e in xrange(d,0,-10):
                        #for f in xrange(e,0,-5):
                            #for g in xrange(f,0,-2):
                                #for h in xrange(h,0,-1):
                                    #count+=1
    #return count

    #This is slow, lots of waste 
    #def coin_recur(amt,prev_coin):
        #if amt == 0:
            #return 1
        #elif amt < 0:
            #return 0
        #else:
            #hits = map(lambda y: coin_recur(amt-y,y),filter(lambda x: x >= prev_coin and amt-x >= 0,purse))
            #return sum(hits)
    #return sum(map(lambda x : coin_recur(initial_amount-x,x),purse))

def ex32():
    totals = set() 
    ref_digits=list(xrange(1,10))
    for a in xrange(1,int(1e5)):
        for b in xrange(1,int(1e5)):
            digit_sum = len(str(a*b)) + len(str(a)) + len(str(b))
            if digit_sum > 9:
                break
            c = a*b
            digits = []
            for x in (a,b,c):
                digits.extend([int(ch) for ch in str(x)])
            #print 'digits: %s, (a,b,c): %s' % (repr(digits),repr((a,b,c)))
            if sorted(digits) == ref_digits:
                #print (a,b,c) 
                totals.add(c)
    return sum(totals)

def ex33():
    '''The fraction 49/98 is a curious fraction, as an inexperienced mathematician in attempting to simplify it may incorrectly believe that 49/98 = 4/8, which is correct, is obtained by cancelling the 9s.
We shall consider fractions like, 30/50 = 3/5, to be trivial examples.
There are exactly four non-trivial examples of this type of fraction, less than one in value, and containing two digits in the numerator and denominator.
If the product of these four fractions is given in its lowest common terms, find the value of the denominator.'''

    def common_digit(sa,sb):
        for i,a in enumerate(sa):
            if a in sb and i != sb.index(a):
                return int(a)
        return None 

    def remove_digit(a,d):
        l = list(str(a))
        l.remove(str(d))
        return int(l[0])

    n,d = 1,1
    for a in xrange(10,100):
        for b in xrange(a+1,100):
            ca,cb = str(a),str(b)
            cd = common_digit(ca,cb)
            if cd:
                ra = remove_digit(a,cd)
                rb = remove_digit(b,cd)
                if rb != 0 and float(a)/float(b) == float(ra)/float(rb):
                    n *= a
                    d *= b
                    print n,d

    return d / gcd(n,d)

def ex34():
    '''145 is a curious number, as 1! + 4! + 5! = 1 + 24 + 120 = 145.
Find the sum of all numbers which are equal to the sum of the factorial of their digits.
Note: as 1! = 1 and 2! = 2 are not sums they are not included.'''
    def fact(n):
        return 1 if n <= 1 else n * fact(n-1)
    MAX = 200000
    val = 0
    def sum_fact_digits(n):
        return sum(map(lambda x: fact(int(x)),list(str(n))))
    for x in xrange(3,MAX+1):
        if x == sum_fact_digits(x):
            val += x 
            print val
    return val

def ex35():
    gp = gen_prime()
    blacklist = set()
    def get_rotations(n):
        sn = str(n)
        for n in xrange(len(sn)):
            yield int(sn[n:] + sn[:n])
    primes = list(itertools.takewhile(lambda x : x < 1000000,gp))
    primeset = set(primes)
    ct = 0
    for p in primes:
        if p in blacklist:
            continue
        else:
            blacklist.add(p)
            rotations = list(get_rotations(p))
            rotations_prime = p in primeset
            for r in rotations:
                blacklist.add(r)
                if not r in primeset:
                    rotations_prime = False
                    break
                
            if rotations_prime:
                ct += len(set(rotations))
    return ct

def ex36():
    def to_binary_str(x):
        return bin(x)[2:]
    def palindrome(s):
        return s == s[::-1]
    ct = 0
    for i in xrange(1,1000000):
        si = str(i)
        bs = to_binary_str(i)
        if '0' == bs[0]:
            continue
        if palindrome(si) and palindrome(bs):
            ct+=i
    return ct

def ex37():
    '''The number 3797 has an interesting property. Being prime itself, it is possible to continuously remove digits from left to right, and remain prime at each stage: 3797, 797, 97, and 7. Similarly we can work from right to left: 3797, 379, 37, and 3.
Find the sum of the only eleven primes that are both truncatable from left to right and right to left.
NOTE: 2, 3, 5, and 7 are not considered to be truncatable primes.'''

    def get_truncated_nums(n):
        sn = str(n)
        for x in xrange(1,len(sn)):
            yield int(sn[x:])
            yield int(sn[:-x])
    gp = gen_prime()
    p = gp.next()
    primeset = set()
    while p < 10:
        primeset.add(p)
        p = gp.next()
    pp_found = 0
    tps = []
    while pp_found < 11:
        primeset.add(p)
        tc = list(get_truncated_nums(p))
        if reduce(lambda x,y: y in primeset and x,tc,True):
            tps.append(p)
            pp_found+=1
        p = gp.next()
    return sum(tps)

def ex38():
    digits = range(1,10)
    def cp(a,n):
        prod = map(lambda k: k*a,range(1,n+1))
        return int(''.join(map(str,prod)))
    def pandig(x):
        return sorted([int(c) for c in str(x)]) == digits
    max_val = 10 ** 9 
    max_pd = -1
    for i in xrange(1,10 ** 5):
        n = 2
        c = cp(i,n)
        while c < max_val:
            if pandig(c):
                max_pd = max(max_pd,c)
            n+=1
            c = cp(i,n) 
    return max_pd

def ex39():
    '''If p is the perimeter of a right angle triangle with integral length sides, {a,b,c}, there are exactly three solutions for p = 120.{20,48,52}, {24,45,51}, {30,40,50} For which value of p <= 1000, is the number of solutions maximised?'''
    pmap = {}
    squares = set([x**2 for x in range(1,1001)])
    for a in xrange(1,1001):
        for b in xrange(1,a+1):
            if a + b > 1000:
                break
            csq = a**2 + b**2
            if csq in squares:
                c = csq ** 0.5
                if a+b+c < 1000:
                    pmap.setdefault(a+b+c,[]).append((a,b,c))
                    
    return sorted(pmap.items(),key=lambda x: len(x[1]),reverse=True)[:5] 

def ex40():
    '''An irrational decimal fraction is created by concatenating the positive integers:
0.123456789101112131415161718192021...
It can be seen that the 12th digit of the fractional part is 1.
If dn represents the nth digit of the fractional part, find the value of the following expression.
d1  d10  d100  d1000  d10000  d100000  d1000000'''
    def d(x):
        dinc = 1 
        dig = 0
        n = 0
        while dig < x:
            n+=1
            if n >= 10**(dinc):
                dinc+=1
            dig+=dinc
        return int(str(n)[x-dig-1])
    print d(1) * d(10) * d(100) * d(1000) * d(10000) * d(100000) * d(1000000)

def ex41():
    perms = map(lambda x: int(''.join(x)),itertools.permutations(''.join([str(x) for x in xrange(1,8)])))
    perms = sorted(filter(lambda x : x % 2 != 0,perms),reverse=True)
    for p in perms:
        if miller_rabin(p,10):
            return p
def ex42():
    import string
    lookup = dict(zip(string.ascii_uppercase,xrange(1,27)))
    def gen_tri():
        n = 1
        while True:
            yield int(0.5 * n * (n+1))
            n +=1
        
    with open('words.txt') as f:
        words = f.readline().replace('"','').split(',')
    wordsum = lambda w: sum ([lookup[l] for l in w])
    sums = map(wordsum, words)
    max_sum = max(sums)
    print max_sum
    tris = set(itertools.takewhile(lambda x: x <= max_sum,gen_tri()))
    tri_words = filter(lambda x: x in tris,sums)
    return len(tri_words)

def ex43():
    '''The number, 1406357289, is a 0 to 9 pandigital number because it is made up of each of the digits 0 to 9 in some order, but it also has a rather interesting sub-string divisibility property.
Let d1 be the 1st digit, d2 be the 2nd digit, and so on. In this way, we note the following:
d2d3d4=406 is divisible by 2
d3d4d5=063 is divisible by 3
d4d5d6=635 is divisible by 5
d5d6d7=357 is divisible by 7
d6d7d8=572 is divisible by 11
d7d8d9=728 is divisible by 13
d8d9d10=289 is divisible by 17
Find the sum of all 0 to 9 pandigital numbers with this property.'''
    evens = set(['2','4','6','8','0'])
    fives = set(['0','5'])
    primes = (2,3,5,7,11,13,17)
    nums = filter(lambda x : x[0] != '0' and x[3] in evens and x[5] in fives,itertools.permutations(list(''.join(map(str,xrange(0,10))))))
    def prime_divis(xs):
        l = []
        divis = True
        for i in xrange(1,8):
            if int(''.join(xs[i:i+3])) % primes[i-1]:
                return False
        return divis
    divis = filter(prime_divis,nums)
    return sum(map(lambda x: int(''.join(x)),divis))

def ex44():
    '''Pentagonal numbers are generated by the formula, Pn=n(3n1)/2. The first ten pentagonal numbers are:
1, 5, 12, 22, 35, 51, 70, 92, 117, 145, ...
It can be seen that P4 + P7 = 22 + 70 = 92 = P8. However, their difference, 70  22 = 48, is not pentagonal.
Find the pair of pentagonal numbers, Pj and Pk, for which their sum and difference are pentagonal and D = |Pk  Pj| is minimised; what is the value of D?'''
    def gen_p():
        n = 1
        while True:
            yield int(n*(3*n-1)/2)
            n+=1
    pents = []
    gp = gen_p()
    pents.append(gp.next())
    pents.append(gp.next())
    pentset = set(pents)
    done = False
    pi = 1
    while not done:
        for i in xrange(pi-1,-1,-1):
            psum = pents[pi] + pents[i]
            pdiff = pents[pi] - pents[i]
            while pents[-1] < psum:
                pent = gp.next()
                pents.append(pent)
                pentset.add(pent)
            if psum in pentset and pdiff in pentset:
                return pdiff 
        pi+=1

def ex45():
    import itertools
    '''Triangle, pentagonal, and hexagonal numbers are generated by the following formulae:
Triangle	 	Tn=n(n+1)/2	 	1, 3, 6, 10, 15, ...
Pentagonal	 	Pn=n(3n1)/2	 	1, 5, 12, 22, 35, ...
Hexagonal	 	Hn=n(2n1)	 	1, 6, 15, 28, 45, ...
It can be verified that T285 = P165 = H143 = 40755.
Find the next triangle number that is also pentagonal and hexagonal.'''

    def g_gen(f):
        n = 1
        while True:
            yield f(n) 
            n+=1
    start = lambda x: x <= 40755
    gen_tri = itertools.dropwhile(start,g_gen(lambda n: int(n*(n+1)/2)))
    gen_pen = itertools.dropwhile(start,g_gen(lambda n: int(n*(3*n-1)/2)))
    gen_hex = itertools.dropwhile(start,g_gen(lambda n: int(n*(2*n-1))))
    t,p,h=gen_tri.next(),gen_pen.next(),gen_hex.next()
    while True:
        h = gen_hex.next()
        while t < h:
            t = gen_tri.next()
        while p < h:
            p = gen_pen.next()
        if h == p and h == t:
            return h

def ex46():
    '''It was proposed by Christian Goldbach that every odd composite number can be written as the sum of a prime and twice a square.
9 = 7 + 212
15 = 7 + 222
21 = 3 + 232
25 = 7 + 232
27 = 19 + 222
33 = 31 + 212
It turns out that the conjecture was false.
What is the smallest odd composite that cannot be written as the sum of a prime and twice a square?'''
    def gen_odd_comp():
        q = 9
        while True:
            if not miller_rabin(q,k=10):
                yield q
            q += 2

    goc = gen_odd_comp()
    def goldbach(x):
        exp = 1
        y = 2*pow(exp,2)
        while x > y:
            rem = x-y
            if miller_rabin(rem,20):
                return True
            exp+=1
            y = 2*pow(exp,2)
        return False
    while True:
        c = goc.next()
        if not goldbach(c):
            return c

def ex47():
    D = {}
    q = 2
    candidates = []
    while len(candidates) < 4:
        if not q in D:
            D[q] = set([q])
            D[q*q] = set([q])
            D[q+q] = set([q])
            candidates = []
        else:
            for p in D[q]:
                D.setdefault(p + q,set()).add(p)
            if len(D[q]) == 4:
                candidates.append(q)
            else:
                candidates = []
        del D[q]
        q+=1
    return candidates[0]

def ex48():
    lim = pow(10,10)
    return sum([pow(x,x,lim) for x in range(1,1001)]) % lim

def ex49():
    import itertools
    '''The arithmetic sequence, 1487, 4817, 8147, in which each of the terms increases by 3330, is unusual in two ways: (i) each of the three terms are prime, and, (ii) each of the 4-digit numbers are permutations of one another.
There are no arithmetic sequences made up of three 1-, 2-, or 3-digit primes, exhibiting this property, but there is one other 4-digit increasing sequence.
What 12-digit number do you form by concatenating the three terms in this sequence?'''
    def perms(x,y):
        digitsx = sorted([int(c) for c in str(x)])
        digitsy = sorted([int(c) for c in str(y)])
        return digitsx == digitsy
    primes = list(itertools.takewhile(lambda x: x <= 9999,itertools.dropwhile(lambda x: x <= 1487,gen_prime())))
    prime_set = set(primes) 
    i_p = 0
    while True:
        p = primes[i_p]
        i_q = i_p + 1
        r = p + 2*(primes[i_q] - p)
        while r <= primes[-1]: 
            if r in prime_set:
                q = primes[i_q]
                if perms(p,q) and perms(p,r):
                    return int(''.join(map(str,(p,q,r))))
            i_q +=1
            r = p + 2*(primes[i_q]-p)
        i_p +=1

def ex50():
    gp = gen_prime()
    primes = list(itertools.takewhile(lambda x: x < 1000000,gp))
    prime_set = set(primes)
    ip_start = 0
    max_consec,max_val = -1,-1
    while True:
        ip = ip_start
        primesum = 0
        prime_count = 0
        if primes[ip_start] + primes[ip_start+1] >= 1000000:
            break
        while primesum + primes[ip] < 1000000:
            prime_count += 1
            primesum += primes[ip]
            if primesum in prime_set:
                if prime_count > max_consec:
                    max_consec,max_val = prime_count,primesum 
            ip+=1
        ip_start +=1
    return max_val

def ex51():
    '''By replacing the 1st digit of *3, it turns out that six of the nine possible values: 13, 23, 43, 53, 73, and 83, are all prime.
By replacing the 3rd and 4th digits of 56**3 with the same digit, this 5-digit number is the first example having seven primes among the ten generated numbers, yielding the family: 56003, 56113, 56333, 56443, 56663, 56773, and 56993. Consequently 56003, being the first member of this family, is the smallest prime with this property.
Find the smallest prime which, by replacing part of the number (not necessarily adjacent digits) with the same digit, is part of an eight prime value family.'''
    primes = itertools.dropwhile(lambda x: x < 100000,itertools.takewhile(lambda x: x < 1000000,gen_prime()))
    primes = list(primes)
    prime_set = set(primes)

    def get_repeats(x,n=3):
        d = {}
        for i,x in enumerate(x):
            d.setdefault(x,set()).add(i)
        return filter(lambda t: len(t[1]) == n,d.items())

    digits = '0123456789'
    def num_prime_families(x):
        '''Checks the number of prime families'''
        x_s = str(x)
        for prime_digit,repeats in get_repeats(x_s):
            prime_count = 1
            family = set([x])
            for d in digits:
                if d == prime_digit:
                    continue
                masked_num = int(''.join([d if i in repeats else x_s[i] for i in range(6)]))
                if masked_num in prime_set and masked_num not in family:
                    prime_count += 1
                    family.add(masked_num)
            if prime_count == 8:
                return family 
        return None
    for p in primes:
        val = num_prime_families(p)
        if val:
            return p 

def ex52():
    '''It can be seen that the number, 125874, and its double, 251748, contain exactly the same digits, but in a different order.
Find the smallest positive integer, x, such that 2x, 3x, 4x, 5x, and 6x, contain the same digits.'''
    n = 1
    def is_perm(x,sorted_y):
        sorted_x = sorted(list(str(x)))
        return sorted_x == sorted_y 
    while True:
        if len(str(n)) != len(str(6*n)):
            n+=1
            continue
        flag = True
        sorted_n = sorted(list(str(n)))
        for mult in range(2,7):
            if not is_perm(n*mult,sorted_n):
                flag = False
                break
        if flag:
            return n
        n+=1

def ex53():
    '''There are exactly ten ways of selecting three from five, 12345:
123, 124, 125, 134, 135, 145, 234, 235, 245, and 345
In combinatorics, we use the notation, 5C3 = 10.
In general,
nCr =	
n!/r!(n-r)!
,where r  n, n! = n(n1)...321, and 0! = 1.
It is not until n = 23, that a value exceeds one-million: 23C10 = 1144066.
How many, not necessarily distinct, values of  nCr, for 1  n  100, are greater than one-million?'''
    cache = {}
    def fact(n):
        if n in cache:
            return cache[n]
        res = 1 if n <= 1 else n * fact(n-1)
        cache[n] = res
        return res

    def ncr(n,r):
        return fact(n) / (fact(r) * fact(n-r))
    
    count = 0
    for n in xrange(1,101):
        for r in xrange(1,n+1):
            if ncr(n,r) > 1000000:
                count+=1
    return count

def ex54():
    card_vals = ('2','3','4','5','6','7','8','9','T','J','Q','K','A')
    card_lookup = dict(zip(card_vals,range(0,len(card_vals))))
    HIGH_CARD,PAIR,TWO_PAIR,THREE_OF_A_KIND,STRAIGHT,FLUSH,FULL_HOUSE,FOUR_OF_A_KIND,STRAIGHT_FLUSH = tuple(xrange(9))

    def evaluate(hand):
        suit_map = {}
        val_map = {}
        max_val = (-1,-1)

        for val,suit in hand:
            suit_map.setdefault(suit,set()).add(val)
            val_map.setdefault(val,set()).add(suit)
            val_len = len(val_map[val])
            if val_len > max_val[1] or (val_len == max_val[1] and val > max_val[0]):
                max_val = (val,val_len)

        if len(suit_map) == 1:
            vals = suit_map.values()[0]
            min_card = min(vals)
            max_card = max(vals)
            if max_card - min_card == 4:
                return (STRAIGHT_FLUSH,max_card)
            else:
                return (FLUSH,max_card)

        if max_val[1] == 4:
            return (FOUR_OF_A_KIND,max_val[0])

        if max_val[1] == 3:
            if len(val_map) == 2:
                return (FULL_HOUSE,max_val[0]) 
            else:
                return (THREE_OF_A_KIND,max_val[0]) 

        if max_val[1] == 2:
            if len(val_map) == 3:
                return (TWO_PAIR,max_val[0])
            else:
                return (PAIR,max_val[0])

        sorted_vals = sorted(val_map.keys())
        if sorted_vals == range(sorted_vals[0],sorted_vals[-1]+1):
            return (STRAIGHT,max_val[0])

        return (HIGH_CARD,max_val[0])
    
    with open('poker.txt') as f:
        lines = map(lambda x: x.strip().split(),f.readlines())

    p1w = 0
    for l in lines:
        split_l = map(lambda x: (card_lookup[x[0]],x[1]),l)
        hand1,hand2 = split_l[:5],split_l[5:]
        eval1,eval2 = evaluate(hand1),evaluate(hand2)
        if eval1[0] > eval2[0]:
            p1w +=1
        elif eval1[0] == eval2[0]:
            if eval1[1] > eval2[1]:
                p1w += 1
            elif eval1[1] == eval2[1]:
                k = lambda x: x[0]
                h1s,h2s = sorted(hand1,key=k,reverse=True),sorted(hand2,key=k,reverse=True)
                for i in xrange(len(h1s)):
                    if h1s[i][0] == eval1[1]:
                        continue
                    if h1s[i][0] > h2s[i][0]:
                        p1w+=1
                    break
    return p1w

def ex55():
    '''If we take 47, reverse and add, 47 + 74 = 121, which is palindromic.
Not all numbers produce palindromes so quickly. For example,
349 + 943 = 1292,
1292 + 2921 = 4213
4213 + 3124 = 7337
That is, 349 took three iterations to arrive at a palindrome.
Although no one has proved it yet, it is thought that some numbers, like 196, never produce a palindrome. A number that never forms a palindrome through the reverse and add process is called a Lychrel number. Due to the theoretical nature of these numbers, and for the purpose of this problem, we shall assume that a number is Lychrel until proven otherwise. In addition you are given that for every number below ten-thousand, it will either (i) become a palindrome in less than fifty iterations, or, (ii) no one, with all the computing power that exists, has managed so far to map it to a palindrome. In fact, 10677 is the first number to be shown to require over fifty iterations before producing a palindrome: 4668731596684224866951378664 (53 iterations, 28-digits).
Surprisingly, there are palindromic numbers that are themselves Lychrel numbers; the first example is 4994.
How many Lychrel numbers are there below ten-thousand?
NOTE: Wording was modified slightly on 24 April 2007 to emphasise the theoretical nature of Lychrel numbers.
    '''

    def lychrel(n):
        def concat_rev(n):
            ns = str(n)
            return n + int(ns[::-1]) 
        def palindrome(n):
            ns = str(n)
            return ns == ns[::-1]
        i = 0
        while i < 50:
            n = concat_rev(n)
            if palindrome(n):
                return False 
            i += 1
        return True
    ly_ct = 0 
    n = 1
    while n < 10000:
        if lychrel(n):
            ly_ct += 1
        n += 1
    print ly_ct

def ex56():
    '''A googol (10100) is a massive number: one followed by one-hundred zeros; 100100 is almost unimaginably large: one followed by two-hundred zeros. Despite their size, the sum of the digits in each number is only 1.
Considering natural numbers of the form, ab, where a, b  100, what is the maximum digital sum?'''

    max_sum = 0
    for a in range(95,100):
        for b in range(95,100):
            max_sum = max(max_sum,sum(map(int,str(pow(a,b)))))
    return max_sum

def ex57():
    '''It is possible to show that the square root of two can be expressed as an infinite continued fraction.
 2 = 1 + 1/(2 + 1/(2 + 1/(2 + ... ))) = 1.414213...
By expanding this for the first four iterations, we get:
1 + 1/2 = 3/2 = 1.5
1 + 1/(2 + 1/2) = 7/5 = 1.4
1 + 1/(2 + 1/(2 + 1/2)) = 17/12 = 1.41666...
1 + 1/(2 + 1/(2 + 1/(2 + 1/2))) = 41/29 = 1.41379...
The next three expansions are 99/70, 239/169, and 577/408, but the eighth expansion, 1393/985, is the first example where the number of digits in the numerator exceeds the number of digits in the denominator.
In the first one-thousand expansions, how many fractions contain a numerator with more digits than denominator?'''

    l_cache = {}
    def lcm(a,b):
        if (a,b) in l_cache:
            return l_cache[(a,b)]
        r = (a * b) / gcd(a,b)
        l_cache[(a,b)] = r
        return r

    g_cache = {}
    def gcd(a,b):
        if (a,b) in g_cache:
            return g_cache[(a,b)]
        r = a if b == 0 else gcd(b,a % b)
        g_cache[(a,b)] = r
        return r

    def add_rat(x,y):
        n_x,d_x = x
        n_y,d_y = y
        l = lcm(d_x,d_y)
        n_x *= (l / d_x)
        n_y *= (l / d_y)
        return (n_x+n_y,l)

    def red_rat(x):
        n_x,d_x = x
        g_x = gcd(n_x,d_x)
        return (n_x / g_x, d_x / g_x)
        
    def div_rat(x,y):
        n_x,d_x = x
        n_y,d_y = y
        return red_rat((d_y * n_x, n_y * d_x))
   
    cache = {}
    def expand(n):
        one = (1,1)
        two = (2,1)
        def expand_inner(n):
            if n == 1:
                return two 
            if n in cache:
                return cache[n]
            r = add_rat(two,div_rat(one,expand_inner(n-1)))
            cache[n] = r
            return r
        return add_rat(one,div_rat(one,expand_inner(n)))

    ct = 0
    for i in xrange(1,1001):
        n,d = expand(i)
        if len(str(n)) > len(str(d)):
            ct +=1
    return ct

def ex58():
    '''Starting with 1 and spiralling anticlockwise in the following way, a square spiral with side length 7 is formed.
37 36 35 34 33 32 31
38 17 16 15 14 13 30
39 18  5  4  3 12 29
40 19  6  1  2 11 28
41 20  7  8  9 10 27
42 21 22 23 24 25 26
43 44 45 46 47 48 49
It is interesting to note that the odd squares lie along the bottom right diagonal, but what is more interesting is that 8 out of the 13 numbers lying along both diagonals are prime; that is, a ratio of 8/13  62%.
If one complete new layer is wrapped around the spiral above, a square spiral with side length 9 will be formed. If this process is continued, what is the side length of the square spiral for which the ratio of primes along both diagonals first falls below 10%?'''
    primes = 0
    s_primes = set()
    composites = 0
    s_composites = set()
    def get_diags(n):
        if n == 1:
            return [1]
        return map(lambda x: pow(n,2) - (n-1) * x,xrange(4))
    n = 1
    while True:
        diags = get_diags(n)
        for d in diags:
            if miller_rabin(d):
                primes += 1
                s_primes.add(d)
            else:
                composites +=1
                s_composites.add(d)
        if n > 7 and (10 * primes) < (composites + primes):
            break
        n+=2
    return n

def ex59():
    '''Each character on a computer is assigned a unique code and the preferred standard is ASCII (American Standard Code for Information Interchange). For example, uppercase A = 65, asterisk (*) = 42, and lowercase k = 107.
A modern encryption method is to take a text file, convert the bytes to ASCII, then XOR each byte with a given value, taken from a secret key. The advantage with the XOR function is that using the same encryption key on the cipher text, restores the plain text; for example, 65 XOR 42 = 107, then 107 XOR 42 = 65.
For unbreakable encryption, the key is the same length as the plain text message, and the key is made up of random bytes. The user would keep the encrypted message and the encryption key in different locations, and without both "halves", it is impossible to decrypt the message.
Unfortunately, this method is impractical for most users, so the modified method is to use a password as a key. If the password is shorter than the message, which is likely, the key is repeated cyclically throughout the message. The balance for this method is using a sufficiently long password key for security, but short enough to be memorable.
Your task has been made easy, as the encryption key consists of three lower case characters. Using cipher1.txt (right click and 'Save Link/Target As...'), a file containing the encrypted ASCII codes, and the knowledge that the plain text must contain common English words, decrypt the message and find the sum of the ASCII values in the original text.'''
    with open('cipher1.txt') as f:
        enc_bytes = map(int,f.readline().strip().split(','))

    cols = (enc_bytes[0::3],enc_bytes[1::3],enc_bytes[2::3])

    def decrypt(key):
        letters = []
        for i,a in enumerate(enc_bytes):
            c = chr(a ^ key[i % 3])
            letters.append(c)
        return ''.join(letters)

    def get_max_elem(s):
        max_char = (-1,-1)
        s = sorted(s)
        i = 0
        while i < len(s):
            c_init = s[i]
            c_count = 0
            while i < len(s) and (s[i] == c_init):
                c_count +=1
                i+=1
            if c_count > max_char[1]:
                max_char = c_init,c_count
        return max_char[0]

    key = map(lambda x : ord(' ') ^ get_max_elem(x),cols)

    return sum(map(ord,decrypt(key)))

def ex60():
    
    '''The primes 3, 7, 109, and 673, are quite remarkable. By taking any two primes and concatenating them in any order the result will always be prime. For example, taking 7 and 109, both 7109 and 1097 are prime. The sum of these four primes, 792, represents the lowest sum for a set of four primes with this property.
Find the lowest sum for a set of five primes for which any two primes concatenate to produce another prime.'''

t0 = time.time()
print ex59()
print 'Time taken: %s' % repr(time.time()-t0)
