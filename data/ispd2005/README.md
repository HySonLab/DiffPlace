# ISPD 2005 Benchmarks

This directory should contain ISPD 2005 placement benchmarks.

## Download Instructions

### Option 1: From DREAMPlace Repository (Recommended)

Download benchmarks from DREAMPlace:
```bash
git clone https://github.com/limbo018/DREAMPlace.git
cp -r DREAMPlace/benchmarks/ispd2005/* data/ispd2005/
```

**Direct Link**: [DREAMPlace Benchmarks](https://github.com/limbo018/DREAMPlace/tree/master/benchmarks)

### Option 2: From ISPD Contest Website

Visit: [ISPD 2005 Placement Contest](http://www.ispd.cc/contests/05/ispd2005_contest.html)

## Expected Structure

After download, you should have:
```
data/ispd2005/
├── adaptec1/
│   ├── adaptec1.aux
│   ├── adaptec1.nodes
│   ├── adaptec1.nets
│   ├── adaptec1.pl
│   ├── adaptec1.scl
│   └── adaptec1.wts
├── adaptec2/
├── adaptec3/
├── adaptec4/
├── bigblue1/
├── bigblue2/
├── bigblue3/
└── bigblue4/
```

## Benchmarks Summary

| Benchmark | Cells | Movable | Fixed |
|-----------|-------|---------|-------|
| adaptec1 | 211k | 211k | ~500 |
| adaptec2 | 255k | 255k | ~500 |
| adaptec3 | 451k | 451k | ~700 |
| adaptec4 | 496k | 496k | ~700 |
| bigblue1 | 278k | 278k | ~500 |
| bigblue2 | 557k | 557k | ~500 |
| bigblue3 | 1.1M | 1.1M | ~500 |
| bigblue4 | 2.2M | 2.2M | ~500 |
