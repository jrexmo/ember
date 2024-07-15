# Horizontal Fills as a Embroidery Primitive

The first method for producing embroideries that Thorne Wolf and Daniel Bump discussed on 2024-07-14 is the horizontal filling method.

<img src="https://github.com/user-attachments/assets/8397edac-b513-485d-b2d9-d68a9e37ac8d" alt="example horizontal filling" width="1024" />

This method is easy to implement and should product reasonable results when rendering generated patterns to PNG. However, the long thread lines might be problematic for real-world embroidery machines. Some pseudo-code for this method is as follows:

```
func generate_image_palette(image):
    ...
func closest_color(color, palette):
    ...
func fill_horizontal(image):
    palette = generate_image_palette(image)
    for y in range(image.height):
        x_start = 0
        while x < image.width:
            x = x_start
            prev_color = closest_color(image.get_pixel(x, y), palette)
            while x < image.width and closest_color(image.get_pixel(x, y)) == prev_color:
                x += 1
            image[x_start:x, y] = prev_color
            x_start = x
```

Pros:
- Easy to implement.
- Should produce reasonable results.

Cons:
- Long threads can be impractical.
- Results might look odd.