package gans

import (
	"image"
	"image/color"

	"github.com/unixpickle/weakai/neuralnet"
)

const GridSpacing = 1

var GridSpaceColor = color.RGBA{R: 0x80, G: 0x80, B: 0x80, A: 0xff}

// GridSample samples images from a generator and arranges
// them in a grid on an image.
// Tensor images may either have a depth of 1 (grayscale)
// or 3 (RGB).
func GridSample(rows, cols int, gen func() *neuralnet.Tensor3) image.Image {
	if rows == 0 && cols == 0 {
		return image.NewRGBA(image.Rect(0, 0, GridSpacing, GridSpacing))
	}

	tensors := make([]*neuralnet.Tensor3, rows*cols)
	for i := range tensors {
		tensors[i] = gen()
	}

	newWidth := tensors[0].Width*cols + (cols+1)*GridSpacing
	newHeight := tensors[0].Height*rows + (rows+1)*GridSpacing
	img := image.NewRGBA(image.Rect(0, 0, newWidth, newHeight))
	for y := 0; y < newHeight; y++ {
		for x := 0; x < newWidth; x++ {
			img.Set(x, y, GridSpaceColor)
		}
	}

	var idx int
	for y := 0; y < rows; y++ {
		tileY := GridSpacing + y*(tensors[0].Height+GridSpacing)
		for x := 0; x < cols; x++ {
			tileX := GridSpacing + x*(tensors[0].Width+GridSpacing)
			tensor := tensors[idx]
			idx++
			for j := 0; j < tensor.Height; j++ {
				for k := 0; k < tensor.Width; k++ {
					if tensor.Depth == 3 {
						img.SetRGBA(tileX+k, tileY+j, color.RGBA{
							R: uint8(tensor.Get(k, j, 0)*0xff + 0.5),
							G: uint8(tensor.Get(k, j, 1)*0xff + 0.5),
							B: uint8(tensor.Get(k, j, 2)*0xff + 0.5),
							A: 0xff,
						})
					} else {
						img.Set(tileX+k, tileY+j, color.Gray{
							Y: uint8(tensor.Get(k, j, 0)*0xff + 0.5),
						})
					}
				}
			}
		}
	}
	return img
}
